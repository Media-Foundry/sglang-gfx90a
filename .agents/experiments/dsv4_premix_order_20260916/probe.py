"""Isolated baseline pre-mix counters; repeated real rows, not service profiling."""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
roctx=ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
for name in ('roctxProfilerPause','roctxProfilerResume'):
    fn=getattr(roctx,name);fn.argtypes=[ctypes.c_uint64];fn.restype=ctypes.c_int
assert roctx.roctxProfilerPause(0)==0
import torch
import triton
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
assert torch.cuda.get_device_properties(0).gcnArchName.startswith('gfx90a')
root=Path(__file__).resolve().parent;repo=root.parents[2]
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb','hc_ffn_fn')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
m=32767
x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed[:4]]
post=post.view(m,4);comb=comb.view(m,4,4);fn=seed[4].contiguous()
hidden=torch.empty_like(residual);partials=torch.empty((m,64),device='cuda');out=torch.empty((m,24),device='cuda')
_post_combine_fused4[(m,16)](x,residual,post,comb,hidden,partials,num_warps=4)
def call():return premix8_pair[(12,triton.cdiv(m,8))](hidden,fn,partials,out,m,1e-6,num_warps=1)
for _ in range(3):kernel=call()
torch.cuda.synchronize();reference=out.clone()
for _ in range(3):
    assert roctx.roctxProfilerResume(0)==0
    call();torch.cuda.synchronize()
    assert roctx.roctxProfilerPause(0)==0
    assert torch.equal(out.view(torch.int32),reference.view(torch.int32))
result=dict(status='complete',scope=__doc__,m=m,physical_gcd=5,profiled_dispatches=3,replay_byte_exact=True,
    registers=kernel.n_regs,spills=kernel.n_spills,lds_bytes=kernel.metadata.shared,
    sources={str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [*paths,Path(__file__),
        repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_premix_pair.py']})
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
