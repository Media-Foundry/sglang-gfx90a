"""Large-M screen of historical exact four-output post-combine reuse.

Repeats/scales real sampled layer-0 inputs, not a full fresh prefill capture.
No production selector. Preserve H256, four wave64s and original expressions.
"""
import argparse
import json
import os
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.layernorm.mhc import mhc_post_combine_rms_triton


@triton.jit
def fused4(x,residual,post,comb,out,partials):
    token=tl.program_id(0);block=tl.program_id(1)
    h=block*256+tl.arange(0,256)
    xv=tl.load(x+token*4096+h).to(tl.float32)
    base=residual+token*16384+h
    r0=tl.load(base).to(tl.float32)
    r1=tl.load(base+4096).to(tl.float32)
    r2=tl.load(base+8192).to(tl.float32)
    r3=tl.load(base+12288).to(tl.float32)
    for hc in tl.static_range(4):
        pv=tl.load(post+token*4+hc)
        # Production contract is comb[input_channel, output_channel].
        cb=comb+token*16+hc
        c0=tl.load(cb);c1=tl.load(cb+4);c2=tl.load(cb+8);c3=tl.load(cb+12)
        acc=pv*xv
        acc+=c0*r0+c1*r1+c2*r2+c3*r3
        tl.store(out+token*16384+hc*4096+h,acc)
        rounded=acc.to(tl.bfloat16).to(tl.float32)
        sq=tl.sum(rounded*rounded,0)
        tl.store(partials+(token*4+hc)*16+block,sq)


def timed(fn):
    start,end=torch.cuda.Event(True),torch.cuda.Event(True)
    start.record()
    for _ in range(5):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)/5


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--large-mutations',type=int,default=3)
p.add_argument('--sizes',type=int,nargs='+',default=[128,8192,32768])
p.add_argument('--runtime',action='store_true',help='Test the integrated dispatch and runtime kernel')
args=p.parse_args();assert not args.output.exists()
if args.runtime:
    assert os.environ.get('SGLANG_DSV4_PREFILL_POST_FUSED4')=='1'
    from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4 as fused4
    from sglang.srt.layers.dsv4_prefill_experiments import _post_reuse
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
root=Path(__file__).resolve().parent
source=root.parent/'dsv4_input_identity_20260914'/'trace-B1'
names=('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')
seeds=[torch.load(source/f'layer_0_rank_0_{name}.pt',map_location='cuda',weights_only=True) for name in names]
assert len({len(t) for t in seeds})==1
torch.manual_seed(20260915)
# Asymmetric comb fixture catches input/output-axis transposition explicitly.
fx=torch.zeros((1,4096),device='cuda',dtype=torch.bfloat16)
fr=torch.tensor([1,2,4,8],device='cuda',dtype=torch.bfloat16).view(1,4,1).expand(1,4,4096).contiguous()
fp=torch.zeros((1,4),device='cuda',dtype=torch.float32)
fc=torch.arange(1,17,device='cuda',dtype=torch.float32).view(1,4,4)
fo=torch.empty_like(fr);fs=torch.empty((1,64),device='cuda',dtype=torch.float32)
fused4[(1,16)](fx,fr,fp,fc,fo,fs,num_warps=4)
expected=torch.tensor([151,166,181,196],device='cuda',dtype=torch.bfloat16).view(1,4,1).expand_as(fo)
assert torch.equal(fo,expected)
assert torch.equal(fs,expected.float().square().view(1,4,16,256).sum(-1).view(1,64))
result=dict(scope=__doc__,runtime=args.runtime,seed_rows=len(seeds[0]),asymmetric_comb_fixture=True,results=[])
for m in args.sizes:
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seeds]
    assert x.shape==(m,4096) and residual.shape==(m,4,4096)
    post=post.view(m,4);comb=comb.view(m,4,4)
    out=torch.empty_like(residual);partials=torch.empty((m,64),dtype=torch.float32,device='cuda')
    def control():return mhc_post_combine_rms_triton(x,residual,post,comb)
    def candidate():
        global out,partials
        if args.runtime:
            token=_post_reuse.set(True)
            try:
                out,partials=mhc_post_combine_rms_triton(x,residual,post,comb)
            finally:
                _post_reuse.reset(token)
            return out,partials
        fused4[(m,16)](x,residual,post,comb,out,partials,num_warps=4)
        return out,partials
    a,b=control();c,d=candidate()
    exact=torch.equal(a,c) and torch.equal(b,d)
    mutation_count=100 if m==128 else args.large_mutations
    for _ in range(mutation_count):
        x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.98,1.02))
        residual.mul_(torch.empty((m,4,1),device='cuda',dtype=residual.dtype).uniform_(.98,1.02))
        post.mul_(torch.empty_like(post).uniform_(.98,1.02))
        comb.mul_(torch.empty_like(comb).uniform_(.98,1.02))
        a,b=control();c,d=candidate()
        exact=exact and torch.equal(a,c) and torch.equal(b,d)
    replay_exact=None
    if m==128:
        graph=torch.cuda.CUDAGraph()
        candidate();torch.cuda.synchronize()
        with torch.cuda.graph(graph):candidate()
        for _ in range(1000):graph.replay()
        torch.cuda.synchronize()
        replay_exact=torch.equal(a,out) and torch.equal(b,partials)
    samples={'A':[],'B':[]}
    for _ in range(3):
        for name,func in [('A',control),('B',candidate),('B',candidate),('A',control)]:
            samples[name].append(timed(func))
    row=dict(m=m,mutation_count=mutation_count,all_exact=exact,
             graph_1000_exact=replay_exact,
             final_out_mismatch=int((a!=c).sum()),final_partials_mismatch=int((b!=d).sum()),
             samples_ms=samples,median_ms={n:median(s) for n,s in samples.items()})
    row['speedup']=row['median_ms']['A']/row['median_ms']['B']
    result['results'].append(row)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    del x,residual,post,comb,out,partials,a,b,c,d
    torch.cuda.empty_cache()
