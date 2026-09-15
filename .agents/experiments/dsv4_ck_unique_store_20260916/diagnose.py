"""Locate the large-M Set mismatch; compare explicit versus historical ABI."""
import hashlib,importlib,importlib.util,json,os
from pathlib import Path
import torch
import aiter
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module
assert os.environ['HIP_VISIBLE_DEVICES']=='6'
root=Path(__file__).resolve().parent;target=root/'diagnose.json';assert not target.exists()
b=json.loads((root/'build-ec7a051f9177/manifest.json').read_text())
spec=importlib.util.spec_from_file_location(b['name'],b['module']);unique=importlib.util.module_from_spec(spec);spec.loader.exec_module(unique)
original=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
fixture=root.parent/'dsv4_c16_real_ck_replay_20260915/capture-v2/fixture'
meta=json.loads((fixture/'manifest.json').read_text());data={}
for name in ('stage2_input','weight13','weight2','sorted_token_ids','sorted_expert_ids','sorted_weights','num_valid_ids'):
    p=fixture/(name+'.pt');assert hashlib.sha256(p.read_bytes()).hexdigest()==meta[name]['file_sha256']
    data[name]=torch.load(p,map_location='cpu',weights_only=True).cuda()
m,t,k=data['stage2_input'].shape
ids=torch.empty_like(data['sorted_token_ids']);fixed_slot_module().remap(data['sorted_token_ids'],data['num_valid_ids'],ids,m)
inter=data['stage2_input'].view(m*t,1,k)
partial=torch.full((m*t,4096),float('nan'),device='cuda');ref=torch.zeros_like(partial)
unique.stage2(inter,data['weight2'],ids,data['sorted_expert_ids'],data['num_valid_ids'],data['sorted_weights'],partial)
base=(inter,data['weight13'],data['weight2'],ids,data['sorted_expert_ids'],data['num_valid_ids'],ref,1,'',None,None,64,data['sorted_weights'],0,1)
def delta(x,y):
    mask=(x!=y);bad=mask.any(1).nonzero().flatten();first=[]
    for row in bad[:8].tolist():
        cols=mask[row].nonzero().flatten()[:4].tolist()
        first.append(dict(virtual_row=row,token=row//6,slot=row%6,cols=cols,
            x=x[row,cols].tolist(),y=y[row,cols].tolist()))
    return dict(changed_elements=int(mask.sum()),changed_rows=len(bad),first=first,
        last_rows=bad[-8:].tolist(),max_abs=float((x-y).abs().max()),finite=bool(torch.isfinite(x).all()))
original.ck_moe_stage2(*base,False)
record=dict(historical_abi=delta(partial,ref))
ref.zero_();original.ck_moe_stage2(*base,1,False,None)
record['explicit_abi']=delta(partial,ref)
target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
