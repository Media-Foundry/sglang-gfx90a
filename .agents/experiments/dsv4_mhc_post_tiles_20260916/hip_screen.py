"""Explicit observed FP32 arithmetic + DPP contract, HIP wave-owned H256."""
import hashlib
import json
import os
from pathlib import Path
import statistics
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
import torch
import triton
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4
root=Path(__file__).resolve().parent;repo=root.parents[2];target=root/'hip-screen.json';assert not target.exists()
mod=load_jit('dsv4_post_wave_screen',cuda_files=[str(root/'exact_post.cuh')],
    cuda_wrappers=[('run','sglang::ExactPostScreen::run')],
    extra_include_paths=[str(repo/'python/sglang/kernels/jit/csrc')],
    extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
report=dict(status='running',scope=__doc__,physical_gcd=4,cases=[],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__),root/'exact_post.cuh']})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
torch.manual_seed(16092026)
for m in (17,8192,32767):
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed]
    post=post.view(m,4);comb=comb.view(m,4,4)
    a=torch.empty_like(residual);b=torch.empty_like(a);sa=torch.empty((m,64),device='cuda');sb=torch.empty_like(sa)
    def baseline():_post_combine_fused4[(m,16)](x,residual,post,comb,a,sa,num_warps=4)
    def candidate():mod.run(x,residual,post,comb,b,sb)
    checks=[]
    for i in range(10):
        if i:
            x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
            residual.mul_(torch.empty((m,4,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
            post.mul_(.999);comb.mul_(1.001)
        b.fill_(float('nan'));sb.fill_(float('nan'));baseline();candidate()
        checks.append(dict(residual_exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
            rms_exact=torch.equal(sa.view(torch.int32),sb.view(torch.int32)),
            residual_max_abs=float((a.float()-b.float()).abs().max()),rms_max_abs=float((sa-sb).abs().max())))
    graphs={}
    for name,fn in [('A',baseline),('B',candidate)]:
        for _ in range(3):fn()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs[name]=g
    samples=[]
    for cycle in range(3):
        for arm in ('A','B','B','A'):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):graphs[arm].replay()
            end.record();end.synchronize();samples.append(dict(cycle=cycle,arm=arm,ms=start.elapsed_time(end)/5))
    med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
    record=dict(m=m,checks=checks,samples=samples,median_ms=med,all_exact=all(c['residual_exact'] and c['rms_exact'] for c in checks))
    report['cases'].append(record);save();print(m,record['all_exact'],checks[0],med,flush=True)
    del graphs,g,x,residual,post,comb,a,b,sa,sb
    torch.cuda.empty_cache()
report['status']='complete';save()
