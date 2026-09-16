"""Real layer0 QKV input/weights: shape and row-order stability, component cost.

Only this isolated process chooses a PyTorch BLAS preference. The explicit
rocBLAS candidate owns its own handle; no service or production selector edits.
"""
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import torch.nn.functional as F
import triton
from sglang.kernels.ops.attention.dsv4.gfx90a_rocblas_linear import Linear
from sglang.kernels.ops.debug.dsv4_prefill_qkv import _project

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'qkv-shape-oracle.json';assert not target.exists()
torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
for arm in ('mixed','serial'):
    assert not json.loads((root/f'trace-v3-{arm}'/f'P32k-tail-policy-trace-v3-{arm}.stop.json').read_text())['remaining']
def sample(arm,stage):
    d=root/f'trace-v2-{arm}'/'trace';stem='layer-0-rank-0-'
    meta=torch.load(d/(stem+'metadata.pt'),weights_only=False)['metadata']
    record=json.loads((d/(stem+stage+'.json')).read_text())
    path=d/record['values_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==record['file_sha256']
    raw=torch.load(path,weights_only=False)['samples']
    selected=[meta['sampled_rows'].index(i) for i in range(64)]
    return raw[selected].contiguous().view(getattr(torch,record['dtype'].removeprefix('torch.'))).cuda()
seed=sample('mixed','prepare_full_input')
assert torch.equal(seed,sample('serial','prepare_full_input'))
weight_path=repo/'.agents/experiments/dsv4_input_identity_20260914/trace-B1/layer_0_rank_0_projection_wqkv_a.pt'
weight=torch.load(weight_path,weights_only=True).cuda()
digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
assert digest(weight)==json.loads((root/'trace-v2-mixed/trace/layer-0-rank-0-projection_wqkv_a.json').read_text())['sha256']
torch.backends.cuda.preferred_blas_library('cublaslt')
rocblas=Linear()
def fixed(x):
    y=torch.empty((len(x),1536),device=x.device,dtype=x.dtype)
    _project[(triton.cdiv(len(x),128),12)](x,weight,y,len(x),num_warps=8,num_stages=2)
    return y
ops={'torch_hipblaslt':lambda x:F.linear(x,weight),'explicit_rocblas':lambda x:rocblas(x,weight),'fixed_triton':fixed}
sources=[Path(__file__),weight_path,repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py',
         repo/'python/sglang/kernels/ops/debug/dsv4_prefill_qkv.py']
report=dict(status='running',scope=__doc__,weight_sha256=digest(weight),input_sha256=digest(seed),cases=[],
            sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def compare(x,y):
    assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
    delta=(x.double()-y.double()).abs()
    return dict(exact=torch.equal(x,y),different_elements=int((x!=y).sum()),
                max_abs=float(delta.max()),mean_abs=float(delta.mean()))
save()
try:
    for mutation in range(3):
        small=seed if mutation==0 else (seed.float()*(1-mutation*.011)).bfloat16()
        small=small.contiguous();ref64=small.double()@weight.double().t()
        small_outputs={name:op(small) for name,op in ops.items()}
        for m in (64,65,8192,32576):
            x=small.repeat((m+63)//64,1)[:m].contiguous()
            permutation=torch.arange(m-1,-1,-1,device='cuda')
            xp=x[permutation].contiguous();item=dict(mutation=mutation,m=m,backends={})
            for name,op in ops.items():
                y=op(x);yp=op(xp)
                metrics=dict(shape=compare(y[:64],small_outputs[name]),
                             row_order=compare(yp[permutation],y),fp64=compare(y[:64],ref64))
                if mutation==0:
                    metrics['service_small']=compare(y[:64],sample('serial','prepare_qkv_a'))
                    metrics['service_large']=compare(y[:64],sample('mixed','prepare_qkv_a'))
                item['backends'][name]=metrics
            samples={name:[] for name in ops}
            # Both forward and reversed order each round; all allocations included.
            for _ in range(3):
                for name in [*ops,*reversed(ops)]:
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();out=ops[name](x);end.record();end.synchronize()
                    samples[name].append(start.elapsed_time(end))
            item['median_ms']={name:statistics.median(values) for name,values in samples.items()}
            report['cases'].append(item);save();print('CASE',mutation,m,item,flush=True)
            del x,xp,permutation,y,yp,out
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
finally:
    rocblas.close()
