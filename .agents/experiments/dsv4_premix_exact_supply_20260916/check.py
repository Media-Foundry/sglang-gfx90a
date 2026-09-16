"""Check hypothesized production arithmetic before timing cooperative supply."""
import json
import os
import sys
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
import triton
from module import load
from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair
constant='--constant-rms' in sys.argv
root=Path(__file__).resolve().parent;target=root/('tail-constant-check.json' if constant else 'tail-check.json');assert not target.exists()
mod=load();torch.manual_seed(2026091606)
report=[]
for m in [1,17,65]:
    x=torch.randn((m,16384),device='cuda',dtype=torch.bfloat16)
    fn=torch.randn((24,16384),device='cuda')*.02
    rms=x.float().view(m,64,256).square().sum(-1).contiguous()
    if constant:rms.fill_(256.)
    a=torch.empty((m,24),device='cuda');b=torch.empty_like(a)
    premix8_pair[(12,triton.cdiv(m,8))](x,fn,rms,a,m,1.e-6,num_warps=1)
    for n in ['d0','d1','s2','s4']:
        b.fill_(float('nan'));getattr(mod,n)(x,fn,rms,b)
        result=dict(m=m,name=n,exact=torch.equal(a.view(torch.int32),b.view(torch.int32)),max_abs=(a-b).abs().max().item(),
            mismatches_by_column=(a.view(torch.int32)!=b.view(torch.int32)).sum(0).tolist(),
            mismatches_by_row=(a.view(torch.int32)!=b.view(torch.int32)).sum(1).tolist())
        print(result,flush=True);report.append(result)
        target.write_text(json.dumps(report,indent=2)+'\n')
