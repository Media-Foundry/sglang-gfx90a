"""Grouped post tile ABBA with independent residual and RMS byte checks."""
import hashlib
import json
import os
from pathlib import Path
import statistics
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
import torch
import triton
from candidate import grouped_post
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'screen.json';assert not target.exists()
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{name}.pt' for name in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')]
seed=[torch.load(path,map_location='cuda',weights_only=True) for path in paths]
torch.manual_seed(16092026)
report=dict(status='running',scope='Repeated/scaled real layer0 sampled rows; not independent live full-M inputs or E2E',
    physical_gcd=4,sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [*paths,root/'candidate.py',Path(__file__),
    repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_post_fused4.py']},cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def byte_equal(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
configs=[(1,1),(1,2),(1,4),(2,4),(4,4),(8,4),(4,8)]
for m in (17,8192,32767):
    x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed]
    post=post.view(m,4);comb=comb.view(m,4,4)
    a=torch.empty_like(residual);b=torch.empty_like(residual)
    sa=torch.empty((m,64),device='cuda',dtype=torch.float32);sb=torch.empty_like(sa)
    case=dict(m=m,candidates=[])
    def baseline():return _post_combine_fused4[(m,16)](x,residual,post,comb,a,sa,num_warps=4)
    for group,warps in configs:
        def candidate():return grouped_post[(m,16//group)](x,residual,post,comb,b,sb,group,num_warps=warps)
        compiled=candidate()
        record=dict(group=group,warps=warps,registers=compiled.n_regs,spills=compiled.n_spills,
            lds_bytes=compiled.metadata.shared,checks=[])
        for mutation in range(5):
            if mutation:
                x.mul_(torch.empty((m,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
                residual.mul_(torch.empty((m,4,1),device='cuda',dtype=residual.dtype).uniform_(.97,1.03))
                post.mul_(.999);comb.mul_(1.001)
            b.fill_(float('nan'));sb.fill_(float('nan'))
            baseline();candidate()
            record['checks'].append(dict(residual_exact=byte_equal(a,b),rms_exact=byte_equal(sa,sb),
                residual_max_abs=float((a.float()-b.float()).abs().max()),rms_max_abs=float((sa-sb).abs().max())))
        record['all_exact']=all(c['residual_exact'] and c['rms_exact'] for c in record['checks'])
        graphs={}
        for name,fn in [('A',baseline),('B',candidate)]:
            for _ in range(3):fn()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):fn()
            graphs[name]=graph
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):graphs[arm].replay()
                end.record();end.synchronize()
                samples.append(dict(arm=arm,cycle=cycle,ms=start.elapsed_time(end)/5))
        centers={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
        record.update(samples=samples,median_ms=centers,speedup=centers['A']/centers['B'])
        case['candidates'].append(record)
        print(m,group,warps,record['all_exact'],centers,flush=True)
        del graphs,graph
    report['cases'].append(case);save()
    del x,residual,post,comb,a,b,sa,sb
    torch.cuda.empty_cache()
report['status']='complete';save()
