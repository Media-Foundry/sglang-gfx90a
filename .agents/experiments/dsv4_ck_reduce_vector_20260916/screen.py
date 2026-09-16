"""Isolated fixed-order vec4 reducer oracle and graph ABBA. No service changes."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--integrated',action='store_true')
args=parser.parse_args()
assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='6'
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module
assert torch.cuda.device_count()==1
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
root=Path(__file__).resolve().parent
mod=load_jit('dsv4_reduce_vec4_screen',cuda_files=[str(root/'vector.cuh')],
    cuda_wrappers=[('bf16','sglang::ReduceVectorScreen::bf16'),('fp32','sglang::ReduceVectorScreen::fp32')],
    extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
    extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
refmod=fixed_slot_module()
blocks_to_test=(1664,) if args.integrated else (416,832,1664)
if args.integrated:
    class Integrated:
        @staticmethod
        def bf16(p,o,b):
            assert b==1664
            refmod.reduce_vec4(p,o)
        @staticmethod
        def fp32(p,o,b):
            assert b==1664
            refmod.reduce_vec4_float(p,o)
    mod=Integrated()
torch.manual_seed(190916)
report=dict(scope='Synthetic full-size partials, isolated reducer only; not routed stage or service throughput',
    integrated=args.integrated,
    physical_gcd=6,status='running',sources={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'screen.py',root/'vector.cuh')},cases=[])
def save():args.output.write_text(json.dumps(report,indent=2)+'\n')
def exact(a,b):return torch.equal(a.view(torch.uint8),b.view(torch.uint8))
def timing(graph):
    a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(5):graph.replay()
    b.record();b.synchronize()
    return a.elapsed_time(b)/5
save()
for m in (1,17,8192,32767,36864):
    partial=torch.randn((m,6,4096),device='cuda',dtype=torch.float32)
    case=dict(m=m,checks=[],abba=[])
    for dtype,name in ((torch.bfloat16,'bf16'),(torch.float32,'fp32')):
        out=torch.empty((m,4096),device='cuda',dtype=dtype);ref=torch.empty_like(out)
        refcall=refmod.reduce if dtype==torch.bfloat16 else refmod.reduce_float
        cand=getattr(mod,name)
        for mutation in range(3):
            if mutation:partial.mul_(.999).add_(.001)
            refcall(partial,ref)
            for blocks in blocks_to_test:
                out.fill_(float('nan'));cand(partial,out,blocks)
                ok=exact(out,ref)
                case['checks'].append(dict(dtype=name,mutation=mutation,blocks=blocks,byte_exact=ok))
                assert ok,case['checks'][-1]
        # Graph mutation checks use stable addresses; input must not be baked in.
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):cand(partial,out,blocks_to_test[0])
        for i in range(100):
            if i%10==0:partial.add_(.0001);refcall(partial,ref)
            out.fill_(float('nan'));graph.replay()
            assert exact(out,ref),(m,name,'graph',i)
        if dtype==torch.bfloat16 and m>=8192:
            calls={'scalar':lambda:refcall(partial,ref)}
            calls.update({f'vec4-{b}':lambda b=b:cand(partial,out,b) for b in blocks_to_test})
            graphs={}
            for key,fn in calls.items():
                for _ in range(3):fn()
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):fn()
                graphs[key]=g
            for key in (f'vec4-{b}' for b in blocks_to_test):
                samples=[]
                for cycle in range(3):
                    for arm in ('scalar',key,key,'scalar'):
                        samples.append(dict(cycle=cycle,arm=arm,ms=timing(graphs[arm])))
                med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('scalar',key)}
                case['abba'].append(dict(candidate=key,samples=samples,median_ms=med,speedup=med['scalar']/med[key]))
            del graphs,calls,g
        del graph,out,ref
    case['graph_replays_per_dtype']=100
    report['cases'].append(case);save()
    print('CASE',m,[(v['candidate'],v['median_ms'],v['speedup']) for v in case['abba']],flush=True)
    del partial
    torch.cuda.empty_cache()
report['status']='complete';save()
