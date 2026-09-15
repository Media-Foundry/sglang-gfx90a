"""Post+premix whole boundary screen; repeated real samples, not E2E."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ['HIP_VISIBLE_DEVICES']=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(r.get('process_info'),dict) for g in owners for r in g.get('process_list',[]))
import torch
import triton
from candidate import post_project,finish
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair

root=Path(__file__).resolve().parent
source=root.parent/'dsv4_input_identity_20260914/trace-B1'
names=('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn')
paths=[source/f'layer_0_rank_0_{n}.pt' for n in names]
seeds=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
result=dict(scope=__doc__,status='running',sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest()
    for p in [*paths,Path(__file__),root/'candidate.py']},cases=[])
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
torch.manual_seed(20260915)
for m in (17,128,8192,32768):
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seeds[:4]]
    post=post.view(m,4);comb=comb.view(m,4,4);fn=seeds[4].contiguous()
    assert fn.shape==(24,16384) and fn.dtype==torch.float32
    a=torch.empty_like(residual);b=torch.empty_like(a)
    sa=torch.empty(m,64,device='cuda');sb=torch.empty_like(sa)
    ma=torch.empty(m,24,device='cuda');mb=torch.empty_like(ma)
    partial=torch.empty(m,16,24,device='cuda')
    def baseline():
        _post_combine_fused4[(m,16)](x,residual,post,comb,a,sa,num_warps=4)
        premix8_pair[(12,triton.cdiv(m,8))](a,fn,sa,ma,m,1e-6,num_warps=1)
    resources=[]
    def candidate():
        kernel=post_project[(m,4)](x,residual,post,comb,fn,b,sb,partial,num_warps=4)
        if not resources:resources.append(dict(registers=kernel.n_regs,spills=kernel.n_spills,lds=kernel.metadata.shared))
        finish[(m,24)](partial,sb,mb,num_warps=1)
    checks=[]
    for iteration in range(3):
        if iteration:
            x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
            residual.mul_(torch.empty((m,4,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
        baseline();candidate()
        checks.append(dict(residual_exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),
            squares_exact=torch.equal(sa.view(torch.int32),sb.view(torch.int32)),
            mixes_exact=torch.equal(ma.view(torch.int32),mb.view(torch.int32)),
            mixes_max_abs=float((ma-mb).abs().max())))
    samples={'A':[],'B':[]}
    for _ in range(3):
        for arm,call in [('A',baseline),('B',candidate),('B',candidate),('A',baseline)]:
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(3):call()
            end.record();end.synchronize();samples[arm].append(start.elapsed_time(end)/3)
    centers={a:statistics.median(v) for a,v in samples.items()}
    item=dict(m=m,checks=checks,samples_ms=samples,median_ms=centers,
              speedup=centers['A']/centers['B'],resources=resources,projection_scratch_bytes=partial.numel()*4)
    result['cases'].append(item);save();print(item,flush=True)
result['status']='complete';save()
