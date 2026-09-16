"""Check integrated dispatch versus measured cooperative oracle before TP8 load."""
import hashlib
import json
import os
from pathlib import Path
import sys
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse import premix_reuse4
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_mfma import _finish
from sglang.srt.layers import dsv4_prefill_experiments as scope
root=Path(__file__).resolve().parent;target=root/'integrated.json';assert not target.exists()
sys.path.insert(0,str(root.parent/'dsv4_premix_mfma_20260916'))
from module import load_cooperative
mod=load_cooperative();torch.manual_seed(2026091605)
report=dict(status='running',cases=[])
os.environ['SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA']='1'
for m in [17,8192,32767]:
    x=torch.randn((m,4,4096),device='cuda',dtype=torch.bfloat16)
    fn=torch.randn((24,16384),device='cuda')*.02
    rms=x.float().view(m,64,256).square().sum(-1).contiguous()
    t=scope._mix_pair.set(False)
    try:off=premix_reuse4(x,fn,rms,1.e-6,group_size=8,pair_columns=True)
    finally:scope._mix_pair.reset(t)
    t=scope._mix_pair.set(True)
    try:on=premix_reuse4(x,fn,rms,1.e-6,group_size=8,pair_columns=True)
    finally:scope._mix_pair.reset(t)
    if m<8192:exact=torch.equal(off.view(torch.int32),on.view(torch.int32))
    else:
        buf=torch.empty((16,m,24),device='cuda');ref=torch.empty_like(on)
        mod.s16(x.view(m,16384),fn,buf)
        _finish[(m,)](buf,rms,ref,m,1.e-6,num_warps=1)
        exact=torch.equal(ref.view(torch.int32),on.view(torch.int32))
    assert exact
    report['cases'].append(dict(m=m,oracle_exact=exact,legacy_exact=torch.equal(off.view(torch.int32),on.view(torch.int32))))
    target.write_text(json.dumps(report,indent=2)+'\n');print(report['cases'][-1],flush=True)
    del x,fn,rms,off,on
    torch.cuda.empty_cache()
report['status']='complete';target.write_text(json.dumps(report,indent=2)+'\n')
