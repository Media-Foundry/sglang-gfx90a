"""Capture launched reference/candidate compiler artifacts for arithmetic audit."""
import hashlib
import json
import os
from pathlib import Path
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
import torch
from candidate import grouped_post
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_fused4 import _post_combine_fused4
root=Path(__file__).resolve().parent;output=root/'compiler';output.mkdir(exist_ok=False)
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
x,residual,post,comb=[torch.load(fixture/f'layer_0_rank_0_{n}.pt',map_location='cuda',weights_only=True)[:17].contiguous()
    for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')]
a=torch.empty_like(residual);b=torch.empty_like(a)
sa=torch.empty((17,64),device='cuda');sb=torch.empty_like(sa)
ref=_post_combine_fused4[(17,16)](x,residual,post,comb,a,sa,num_warps=4)
cand=grouped_post[(17,2)](x,residual,post,comb,b,sb,8,num_warps=4)
torch.cuda.synchronize()
record=dict(residual_exact=torch.equal(a.view(torch.uint8),b.view(torch.uint8)),rms_exact=torch.equal(sa.view(torch.int32),sb.view(torch.int32)),artifacts={})
for name,kernel in [('reference',ref),('candidate',cand)]:
    for kind in ('ttir','ttgir','llir','amdgcn'):
        path=output/(name+'.'+kind);path.write_text(kernel.asm[kind]);record['artifacts'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    record[name]=dict(registers=kernel.n_regs,spills=kernel.n_spills,lds=kernel.metadata.shared)
(output/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record,indent=2))
