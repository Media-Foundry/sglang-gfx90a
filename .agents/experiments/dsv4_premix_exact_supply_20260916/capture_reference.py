"""Pin the actual reference object and reduction IR before writing exact HIP."""
import ctypes
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
root=Path(__file__).resolve().parent;target=root/'reference.json';assert not target.exists()
m=32767
x=torch.ones((m,16384),device='cuda',dtype=torch.bfloat16)
fn=torch.ones((24,16384),device='cuda');partials=torch.full((m,64),256.,device='cuda');out=torch.empty((m,24),device='cuda')
k=premix8_pair[(12,triton.cdiv(m,8))](x,fn,partials,out,m,1.e-6,num_warps=1)
torch.cuda.synchronize();assert torch.isfinite(out).all()
for kind in ['ttgir','llir','amdgcn']:(root/f'reference.{kind}').write_text(k.asm[kind])
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32);assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
target.write_text(json.dumps(dict(pci=pci.value.decode(),m=m,registers=k.n_regs,spills=k.n_spills,
    hashes={kind:hashlib.sha256(k.asm[kind].encode()).hexdigest() for kind in ['ttgir','llir','amdgcn']}),indent=2)+'\n')
