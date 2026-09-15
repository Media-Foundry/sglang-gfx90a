"""Pending isolated component screen; forbid overlap with service benchmarks."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--sizes',type=int,nargs='+',default=[17,128,8192,32767,32768])
p.add_argument('--mutations',type=int,default=10)
p.add_argument('--randomize',action='store_true',help='Vary temporary Fn and activations; never writes checkpoint files.')
p.add_argument('--columns', type=int, choices=(3,4), required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
assert args.mutations>0 and all(0<m<=65536 for m in args.sizes)
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
active={r['process_info']['pid'] for g in owners for r in g.get('process_list',[]) if isinstance(r.get('process_info'),dict)}
assert not active, f'GPU occupied; do not benchmark alongside service: {active}'

import torch
import triton
from candidate import premix8_cols3, premix8_cols4
candidate_kernel={3:premix8_cols3,4:premix8_cols4}[args.columns]
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
root=Path(__file__).resolve().parent;repo=root.parents[2]
source=root.parent/'dsv4_input_identity_20260914/trace-B1'
fn_path=source/'layer_0_rank_0_hc_ffn_fn.pt'
x_path=source/'layer_0_rank_0_ffn_mhc_residual.pt'
torch.manual_seed(20260915)
fn=torch.load(fn_path,map_location='cuda',weights_only=True).contiguous()
seed=torch.load(x_path,map_location='cuda',weights_only=True)
files=[fn_path,x_path,Path(__file__),root/'candidate.py',repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py']
result=dict(status='running',columns=args.columns,scope='Captured layer0 residual/Fn expanded to synthetic occupancy; not fresh service activations or E2E.',
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},results=[])
def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
def exact(a,b):return torch.equal(a.view(torch.int32),b.view(torch.int32))
def measure(fn):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/5
save()
for m in args.sizes:
    x=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    partials=x.float().square().reshape(m,64,256).sum(-1)
    a=torch.empty((m,1,24),device='cuda',dtype=torch.float32);b=torch.empty_like(a)
    eps=1e-6
    def control():return premix8_pair[(12,triton.cdiv(m,8))](x,fn,partials,a,m,eps,num_warps=1)
    def candidate():return candidate_kernel[(24//args.columns,triton.cdiv(m,8))](x,fn,partials,b,m,eps,num_warps=1)
    item=dict(m=m,checks=[],samples_ms={'A':[],'B':[]});result['results'].append(item);save()
    for mutation in range(args.mutations):
        if args.randomize:
            fn.normal_(std=.01);x.normal_()
        elif mutation:
            x.mul_(torch.empty((m,1,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
        eps=(1e-6,1e-5,1e-8)[mutation%3]
        partials.copy_(x.float().square().reshape(m,64,256).sum(-1))
        a.fill_(float('nan'));b.fill_(float('nan'));control();candidate()
        check=dict(bits_exact=exact(a,b),max_abs=float((a-b).abs().max()),finite=bool(torch.isfinite(b).all()))
        item['checks'].append(check);save()
        assert check['bits_exact'] and check['finite'],check
    old=a.clone();order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
    x.copy_(x[order]);partials.copy_(partials[order]);control();candidate()
    item['row_permutation_exact']=exact(old,a[inverse]) and exact(a,b)
    assert item['row_permutation_exact'];save()
    eps=1e-6
    control();candidate();assert exact(a,b)
    for _ in range(3):
        for label,call in [('A',control),('B',candidate),('B',candidate),('A',control)]:
            item['samples_ms'][label].append(measure(call))
    item['randomized']=args.randomize
    item['median_ms']={k:statistics.median(v) for k,v in item['samples_ms'].items()}
    item['speedup_percent']=100*(item['median_ms']['A']/item['median_ms']['B']-1)
    item['resources']={}
    for label,call in [('A',control),('B',candidate)]:
        c=call();item['resources'][label]=dict(registers=c.n_regs,spills=c.n_spills,lds=c.metadata.shared,
            hsaco_sha256=hashlib.sha256(c.asm['hsaco']).hexdigest())
    candidate();torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):candidate()
    for _ in range(1000):graph.replay()
    control();assert exact(a,b)
    x.mul_(1.01);partials.copy_(x.float().square().reshape(m,64,256).sum(-1))
    graph.replay();control();assert exact(a,b)
    item['graph_replay_and_mutation_exact']=True;save()
    print(json.dumps({k:v for k,v in item.items() if k not in ('checks','samples_ms')}),flush=True)
    del graph,x,partials,a,b,old,order,inverse
    torch.cuda.empty_cache()
result['status']='complete';save()

