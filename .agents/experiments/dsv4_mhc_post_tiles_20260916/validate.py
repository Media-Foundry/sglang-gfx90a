"""Integrated HIP post: mutations, row permutation, graph replay and comb orientation."""
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
import torch
import triton
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module
root=Path(__file__).resolve().parent;repo=root.parents[2];target=root/'validation.json';assert not target.exists()
mod=post_wave_module();torch.manual_seed(2026091601)
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
sources=[Path(__file__),repo/'python/sglang/kernels/jit/csrc/deepseek_v4/gfx90a_mhc_post_wave.cuh',
    repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_wave.py',repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_fused4.py']
report=dict(status='running',physical_gcd=4,cases=[],sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def eq(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
for m in (1,17,128,8192,32767,32768,65536):
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed]
    post=post.view(m,4);comb=comb.view(m,4,4)
    a=torch.empty_like(residual);b=torch.empty_like(a);sa=torch.empty((m,64),device='cuda');sb=torch.empty_like(sa)
    def baseline():_post_combine_fused4[(m,16)](x,residual,post,comb,a,sa,num_warps=4)
    def candidate():mod.run(x,residual,post,comb,b,sb)
    checks=[]
    for i in range(100):
        if i and i%25==0:
            x.normal_();residual.normal_();post.normal_();comb.normal_()
        elif i:
            x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
            residual.mul_(torch.empty((m,4,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
            post.add_(torch.randn_like(post)*.001);comb.add_(torch.randn_like(comb)*.001)
        b.fill_(float('nan'));sb.fill_(float('nan'));baseline();candidate()
        ok=eq(a,b) and eq(sa,sb)
        checks.append(ok)
        if not ok:
            report.update(status='failed',failure=dict(m=m,mutation=i,residual_exact=eq(a,b),rms_exact=eq(sa,sb)));save();raise AssertionError(report['failure'])
    reversed_inputs=[p.flip(0) for p in (x,residual,post,comb)]
    mod.run(*reversed_inputs,b,sb)
    row_exact=eq(a,b.flip(0)) and eq(sa,sb.flip(0));assert row_exact
    del reversed_inputs
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):candidate()
    replays=1000 if m==128 else 100
    for i in range(replays):
        if i%10==0:x.mul_(.999);baseline()
        b.fill_(float('nan'));sb.fill_(float('nan'));g.replay()
        assert eq(a,b) and eq(sa,sb),(m,'graph',i)
    if m==1:
        x.fill_(2)
        for c in range(4):residual[:,c].fill_(c+1)
        post.copy_(torch.arange(1,5,device='cuda').view(1,4))
        comb.copy_(torch.arange(1,17,device='cuda').view(1,4,4));candidate()
        expected=torch.tensor([92,104,116,128],device='cuda',dtype=torch.bfloat16)
        assert eq(b,expected.view(1,4,1).expand_as(b).contiguous())
        expected_rms=(256*expected.float().square()).view(1,4,1).expand(1,4,16).reshape(1,64).contiguous()
        assert eq(sb,expected_rms)
        report['asymmetric_comb_orientation_exact']=True
    report['cases'].append(dict(m=m,mutations_exact=checks,row_permutation_exact=row_exact,graph_replays_exact=replays));save()
    print('VALIDATED',m,100,replays,flush=True)
    del g,x,residual,post,comb,a,b,sa,sb
    torch.cuda.empty_cache()
report['status']='complete';save()
