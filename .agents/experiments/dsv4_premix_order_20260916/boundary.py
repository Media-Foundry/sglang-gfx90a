"""Full HIP post + pre-mix boundary ABBA for the best isolated CTA-order candidate."""
import hashlib
import json
import os
from pathlib import Path
import statistics
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
from candidate import premix_order
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_wave import post_wave_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
root=Path(__file__).resolve().parent;repo=root.parents[2];target=root/'boundary.json';assert not target.exists()
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
report=dict(status='running',scope='Repeated/scaled sampled real layer0 inputs; full post+premix boundary, not live full-M or service throughput',
    physical_gcd=5,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,Path(__file__),root/'candidate.py',
    repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py']},cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
torch.manual_seed(20260916);postmod=post_wave_module()
for m in (8192,32767):
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
    post=post.view(m,4);comb=comb.view(m,4,4);fn=seed[4].contiguous()
    hidden=torch.empty_like(residual);partials=torch.empty((m,64),device='cuda')
    postmod.run(x,residual,post,comb,hidden,partials)
    a=torch.empty((m,24),device='cuda');b=torch.empty_like(a)
    case=dict(m=m,candidates=[])
    def baseline():
        postmod.run(x,residual,post,comb,hidden,partials)
        return premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,partials,a,m,1e-6,num_warps=1)
    for group in (8,):
        def candidate():
            postmod.run(x,residual,post,comb,hidden,partials)
            return premix_order[(12*triton.cdiv(m,8),)](hidden,fn,partials,b,m,group,triton.cdiv(m,8)%group==0,1e-6,num_warps=1)
        kernel=candidate();checks=[]
        for mutation in range(5):
            if mutation:
                x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
                postmod.run(x,residual,post,comb,hidden,partials)
            b.fill_(float('nan'));baseline();candidate()
            checks.append(torch.equal(a.view(torch.int32),b.view(torch.int32)))
        graphs={}
        for name,call in [('A',baseline),('B',candidate)]:
            for _ in range(3):call()
            g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):call()
            graphs[name]=g
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):graphs[arm].replay()
                end.record();end.synchronize();samples.append(dict(cycle=cycle,arm=arm,ms=start.elapsed_time(end)/5))
        med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
        case['candidates'].append(dict(group=group,checks=checks,all_exact=all(checks),median_ms=med,samples=samples,
            registers=kernel.n_regs,spills=kernel.n_spills,lds_bytes=kernel.metadata.shared))
        print(m,group,all(checks),med,flush=True)
        del g,graphs
    report['cases'].append(case);save()
    del x,residual,post,comb,hidden,partials,a,b
    torch.cuda.empty_cache()
report['status']='complete';save()
