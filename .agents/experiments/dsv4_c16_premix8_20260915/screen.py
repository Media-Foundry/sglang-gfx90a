"""Production four-row versus explicit eight-row pre-mix component.

Repeat/scale captured real residuals to form synthetic occupancy, not fresh
large-M model activations. Include full pre-mix and RMS-partial consumption;
no model precision/selector change and no permanent workspace.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import torch
import triton
from candidate import premix8
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse import (
    _premix_reuse_kernel, premix_reuse4)

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', type=Path, required=True)
p.add_argument('--sizes', type=int, nargs='+', default=[17,128,8192,32768,65536])
p.add_argument('--mutations', type=int, default=3)
p.add_argument('--randomize', action='store_true')
args = p.parse_args()
assert not args.output.exists() and all(m > 0 for m in args.sizes)
assert os.environ.get('HIP_VISIBLE_DEVICES') == '4'
torch.manual_seed(20260915)
root = Path(__file__).resolve().parent
source = root.parent/'dsv4_input_identity_20260914/trace-B1'
fn_path = source/'layer_0_rank_0_hc_ffn_fn.pt'
seed_path = source/'layer_0_rank_0_ffn_mhc_residual.pt'
fn_seed = torch.load(fn_path, map_location='cuda', weights_only=True).contiguous()
seed = torch.load(seed_path, map_location='cuda', weights_only=True)
fn = fn_seed.clone()
result = dict(status='running', scope=__doc__, gpu=4, results=[],
    source_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        fn_path, seed_path, Path(__file__), root/'candidate.py',
        root.parents[2]/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_reuse.py')})

def save():
    args.output.write_text(json.dumps(result,indent=2)+'\n')

def compare(a,b):
    return dict(bits_exact=torch.equal(a.view(torch.int32), b.view(torch.int32)),
        max_abs=float((a-b).abs().max()), mismatches=int((a.view(torch.int32)!=b.view(torch.int32)).sum()),
        finite=bool(torch.isfinite(b).all()))

def measure(func):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(5): func()
    end.record();end.synchronize()
    return start.elapsed_time(end)/5

def metadata(c):
    return dict(registers=c.n_regs, spills=c.n_spills, lds=c.metadata.shared,
        hash=c.hash, hsaco_sha256=hashlib.sha256(c.asm['hsaco']).hexdigest())

save()
for m in args.sizes:
    fn.copy_(fn_seed)
    x=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    x.mul_(torch.empty((m,1,1),device='cuda',dtype=x.dtype).uniform_(.8,1.2))
    partials=x.float().square().view(m,64,256).sum(-1)
    outputs=[torch.empty((m,1,24),device='cuda',dtype=torch.float32) for _ in range(2)]
    eps=1e-6
    def control():
        return _premix_reuse_kernel[(24,triton.cdiv(m,4))](x,fn,partials,outputs[0],m,4,eps,num_warps=1)
    def candidate():
        return premix8[(24,triton.cdiv(m,8))](x,fn,partials,outputs[1],m,eps,num_warps=1)
    for _ in range(3):control();candidate()
    wrapper=premix_reuse4(x,fn,partials,eps)
    assert compare(wrapper,outputs[0])['bits_exact']
    initial=compare(*outputs)
    assert initial['bits_exact'],initial
    samples={'A':[],'B':[]}
    for _ in range(3):
        for label,func in (('A',control),('B',candidate),('B',candidate),('A',control)):
            samples[label].append(measure(func))
    checks=[]
    for mutation in range(args.mutations):
        if args.randomize:
            fn.normal_(std=.01);x.normal_()
        else:
            x.mul_(torch.empty((m,1,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
        eps=(1e-6,1e-5,1e-8)[mutation%3]
        partials.copy_(x.float().square().view(m,64,256).sum(-1))
        outputs[0].fill_(float('nan'));outputs[1].fill_(float('nan'))
        control();candidate();check=compare(*outputs);checks.append(check)
        assert check['bits_exact'] and check['finite'],(m,mutation,check)
    before=[y.clone() for y in outputs]
    order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
    x.copy_(x[order]);partials.copy_(partials[order])
    control();candidate()
    permutations=[compare(a,b[inverse]) for a,b in zip(before,outputs)]
    assert all(c['bits_exact'] for c in permutations),permutations
    graph_checks=None
    if m==128:
        candidate();torch.cuda.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):candidate()
        for _ in range(1000):graph.replay()
        control();graph_checks=compare(*outputs)
        assert graph_checks['bits_exact']
        del graph
    item=dict(m=m,initial=initial,mutations=checks,permutations=permutations,
        samples_ms=samples,median_ms={k:median(v) for k,v in samples.items()},
        resources={'A':metadata(control()),'B':metadata(candidate())},
        resources_eps=eps,timed_eps=1e-6,
        graph_1000=graph_checks,randomized=args.randomize)
    item['gain_percent']=100*(item['median_ms']['A']/item['median_ms']['B']-1)
    result['results'].append(item);save()
    print(json.dumps({k:v for k,v in item.items() if k not in ('mutations','samples_ms')}),flush=True)
    del x,partials,outputs,wrapper,before,order,inverse
    torch.cuda.empty_cache()
result['status']='complete';save()
