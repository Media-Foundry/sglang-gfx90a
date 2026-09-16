"""Small fixed-K QKV geometry screen: exact to fixed128 reference, not old BLAS.

Only real layer0 residual-derived input and checkpoint-identical BF16 weight.
Graph component timing excludes the rest of prepare and service scheduling.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from sglang.kernels.ops.debug.dsv4_prefill_qkv import _project as reference_kernel

@triton.jit(do_not_specialize=['M'])
def project(X,W,Y,M,BM:tl.constexpr,BN:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,128)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(32):
        kk=start*128+k
        a=tl.load(X+m[:,None]*4096+kk[None,:],m[:,None]<M,other=0)
        b=tl.load(W+n[None,:]*4096+kk[:,None])
        acc=tl.dot(a,b,acc)
    tl.store(Y+m[:,None]*1536+n[None,:],acc,m[:,None]<M)

root=Path(__file__).resolve().parent;repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--remaining',action='store_true')
args=parser.parse_args()
target=root/('qkv-fixed-screen-remaining.json' if args.remaining else 'qkv-fixed-screen.json');assert not target.exists()
torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
proof=json.loads((root/'qkv-shape-oracle.json').read_text());assert proof['status']=='complete'
assert all(c['backends']['fixed_triton']['shape']['exact'] and c['backends']['fixed_triton']['row_order']['exact'] for c in proof['cases'])
d=root/'trace-v2-mixed/trace';stem='layer-0-rank-0-'
meta=torch.load(d/(stem+'metadata.pt'),weights_only=False)['metadata']
sample_path=d/(stem+'prepare_full_input.pt')
raw=torch.load(sample_path,weights_only=False)['samples']
seed=raw[[meta['sampled_rows'].index(i) for i in range(64)]].contiguous().view(torch.bfloat16).cuda()
weight_path=repo/'.agents/experiments/dsv4_input_identity_20260914/trace-B1/layer_0_rank_0_projection_wqkv_a.pt'
w=torch.load(weight_path,weights_only=True).cuda()
digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert digest(w)==proof['weight_sha256'] and digest(seed)==proof['input_sha256']
torch.backends.cuda.preferred_blas_library('cublaslt')
geometries=[(128,128,8),(16,128,4),(32,128,4),(64,64,4),(64,128,4),
            (128,64,4),(128,128,4),(64,256,8),(128,256,8),(128,64,8)]
if args.remaining:
    previous=json.loads((root/'qkv-fixed-screen.json').read_text())
    assert previous['status']=='failed'
    assert [tuple(c['geometry']) for c in previous['cases']]==geometries[:7]
    geometries=geometries[7:]
report=dict(status='running',scope=__doc__,weight_sha256=digest(w),input_sha256=digest(seed),cases=[],
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def launch(x,y,geometry):
    bm,bn,waves=geometry
    project[(triton.cdiv(len(x),bm),1536//bn)](x,w,y,len(x),BM=bm,BN=bn,num_warps=waves,num_stages=2)
def reference(x):
    y=torch.empty((len(x),1536),device='cuda',dtype=torch.bfloat16)
    reference_kernel[(triton.cdiv(len(x),128),12)](x,w,y,len(x),num_warps=8,num_stages=2)
    return y
def capture(op):
    for _ in range(3):out=op()
    torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):out=op()
    return graph,out
def time_graph(graph):
    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(8):graph.replay()
    end.record();end.synchronize()
    return start.elapsed_time(end)/40
save()
try:
    for geometry in geometries:
        case=dict(geometry=geometry,checks=[],timings=[])
        try:
            probe=torch.empty((64,1536),device='cuda',dtype=torch.bfloat16)
            launch(seed,probe,geometry);torch.cuda.synchronize()
        except triton.runtime.errors.OutOfResources as error:
            case['rejected']='compiler resource limit'
            case['resource_error']=str(error)
            report['cases'].append(case);save();print('REJECTED',geometry,str(error),flush=True)
            continue
        for m in (64,65,8192,32576):
            x=seed.repeat((m+63)//64,1)[:m].contiguous()
            y=torch.empty((m,1536),device='cuda',dtype=torch.bfloat16)
            for mutation in range(3):
                if mutation:x.mul_(.989)
                launch(x,y,geometry);ref=reference(x)
                permutation=torch.arange(m-1,-1,-1,device='cuda')
                xp=x[permutation].contiguous();yp=torch.empty_like(y)
                launch(xp,yp,geometry)
                small=reference(x[:64].contiguous())
                check=dict(m=m,mutation=mutation,reference_exact=torch.equal(y,ref),
                           shape_exact=torch.equal(y[:64],small),row_order_exact=torch.equal(yp[permutation],y))
                case['checks'].append(check)
                del permutation,xp,yp,ref,small
            if not all(c['reference_exact'] and c['shape_exact'] and c['row_order_exact'] for c in case['checks']):
                case['rejected']='numerical order changed';break
            candidate,out=capture(lambda:(launch(x,y,geometry),y)[1])
            control,base=capture(lambda:F.linear(x,w))
            samples={'control':[],'candidate':[]}
            for _ in range(4):
                for name,g in (('control',control),('candidate',candidate),('candidate',candidate),('control',control)):
                    samples[name].append(time_graph(g))
            case['timings'].append(dict(m=m,samples_ms=samples,
                median_ms={k:statistics.median(v) for k,v in samples.items()}))
            del candidate,control,out,base,x,y
        report['cases'].append(case);save();print('GEOMETRY',geometry,case.get('rejected'),case['timings'],flush=True)
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
