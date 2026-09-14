"""Isolated large-M FP32 pre-mix screen; no service selector or precision change.

Uses real layer-0 FP32 Fn and repeated/scaled sampled real residual rows. This
is a synthetic occupancy screen, not a fresh model trace or E2E quality oracle.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import time

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import gfx90a_mhc_pre_mix_from_partials_triton


@triton.jit
def scale_dot(dot, partials, output, EPS: tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,64)
    sq=tl.sum(tl.load(partials+row*64+k),0)
    scale=tl.rsqrt(sq/16384+EPS)
    cols=tl.arange(0,32)
    val=tl.load(dot+row*24+cols,cols<24,0)
    tl.store(output+row*24+cols,val*scale,cols<24)


def timed(fn):
    start,end=torch.cuda.Event(True),torch.cuda.Event(True)
    start.record()
    for _ in range(3):fn()
    end.record();end.synchronize()
    return start.elapsed_time(end)/3


def errors(a,b):
    da,db=a.float(),b.float();delta=(da-db).abs()
    return dict(exact=torch.equal(a,b),finite=bool(torch.isfinite(b).all()),
                max_abs=float(delta.max()),mean_abs=float(delta.mean()),
                relative_l2=float(torch.linalg.vector_norm(da-db)/torch.linalg.vector_norm(da)),
                cosine=float(torch.nn.functional.cosine_similarity(da.flatten(),db.flatten(),dim=0)))


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
assert not os.environ.get('SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR')
os.environ['SGLANG_DSV4_GFX90A_MHC_BLOCK_K']='1024'
torch.backends.cuda.matmul.allow_tf32=False
torch.manual_seed(20260915)
root=Path(__file__).resolve().parent
source=root.parent/'dsv4_input_identity_20260914'/'trace-B1'
weight_path=source/'layer_0_rank_0_hc_ffn_fn.pt'
fn=torch.load(weight_path,map_location='cuda',weights_only=True).contiguous()
seed=torch.load(source/'layer_0_rank_0_ffn_mhc_residual.pt',map_location='cuda',weights_only=True)
assert fn.dtype==torch.float32 and seed.dtype==torch.bfloat16
result=dict(scope=__doc__,hip_visible_devices='4',fn_sha256=hashlib.sha256(weight_path.read_bytes()).hexdigest(),
            seed_rows=seed.shape[0],weight_dtype=str(fn.dtype),results=[])
for m in (8192,32768):
    x=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    x.mul_(torch.empty((m,1,1),device='cuda',dtype=x.dtype).uniform_(.8,1.2))
    xf=torch.empty((m,16384),device='cuda',dtype=torch.float32)
    raw=torch.empty((m,24),device='cuda',dtype=torch.float32)
    out=torch.empty((m,1,24),device='cuda',dtype=torch.float32)
    partials=x.float().square().view(m,64,256).sum(-1).contiguous()
    def control():return gfx90a_mhc_pre_mix_from_partials_triton(x,fn,partials,1e-6)
    def candidate():
        xf.copy_(x.flatten(1))  # Included in timing and memory, not hidden.
        torch.mm(xf,fn.t(),out=raw)
        scale_dot[(m,)](raw,partials,out,1e-6,num_warps=1)
        return out
    for _ in range(3):control();candidate()
    a=control().clone();b=candidate().clone()
    initial=errors(a,b);assert initial['finite']
    repeats=[]
    for _ in range(10):repeats.append(torch.equal(b,candidate()))
    samples={'A':[],'B':[]}
    for _ in range(3):
        for name,func in [('A',control),('B',candidate),('B',candidate),('A',control)]:
            samples[name].append(timed(func))
    mutations=[]
    for _ in range(3):
        x.mul_(torch.empty((m,1,1),device='cuda',dtype=x.dtype).uniform_(.97,1.03))
        partials.copy_(x.float().square().view(m,64,256).sum(-1))
        mutations.append(errors(control().clone(),candidate().clone()))
    # Fixed input, changed row placement: compare after inverse permutation.
    reference=candidate().clone();order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
    x.copy_(x[order]);partials.copy_(partials[order])
    perm_error=errors(reference,candidate()[inverse])
    row=dict(m=m,fp32_activation_workspace_bytes=xf.numel()*xf.element_size(),
             samples_ms=samples,median_ms={k:median(v) for k,v in samples.items()},
             initial=initial,mutations=mutations,fixed_replay_exact=all(repeats),
             row_permutation=perm_error)
    row['speedup']=row['median_ms']['A']/row['median_ms']['B']
    result['results'].append(row)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(row),flush=True)
    del x,xf,raw,out,partials,a,b,reference,order,inverse
    torch.cuda.empty_cache()
