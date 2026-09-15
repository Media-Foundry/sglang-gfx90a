"""Selected real-row FP64 oracle + complete stage2 ABBA, not service timing."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics

import torch

torch.set_num_threads(4)

assert os.environ.get('HIP_VISIBLE_DEVICES')=='6'
root=Path(__file__).resolve().parent; fixture=root/'capture/fixture'
assert not (root/'reference.json').exists()
meta=json.loads((fixture/'manifest.json').read_text())
import aiter
from aiter.ops.shuffle import shuffle_weight
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import _logical_a16w4_scales
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import ck_fixed_slot_stage2

def load(name):
    path=fixture/(name+'.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
    x=torch.load(path,weights_only=True,map_location='cpu')
    return x

cpu={name:load(name) for name in ('weight13','weight2','stage2_input','sorted_token_ids',
    'sorted_expert_ids','num_valid_ids','sorted_weights','topk_ids','topk_weights',
    'raw_w2','raw_s2','stage2_out')}
rows=[2555,3414,4513]
w2=cpu['weight2'];e,n,k=w2.shape
logical=(w2.reshape(e,n//16,k//32,4,16,8).permute(0,1,4,2,3,5).contiguous().view_as(w2)
         if meta['weight2']['is_shuffled'] else w2)
assert torch.equal(shuffle_weight(logical).view(torch.uint8),w2.view(torch.uint8)) if meta['weight2']['is_shuffled'] else True
scales=(_logical_a16w4_scales(cpu['raw_s2'],e,n,k//32,gate_up=False)
        if meta['scales_shuffled'] else cpu['raw_s2'].view(e,n,k//32))
experts=cpu['topk_ids'][rows].unique().long()
# Production decodes through signed INT8 code units, so E2M1 -0 becomes +0.
# This preserves its byte contract; both zero signs have the same real value.
table=torch.tensor([0,.5,1,1.5,2,3,4,6,0.,-.5,-1,-1.5,-2,-3,-4,-6],dtype=torch.float32)
for expert in experts:
    packed=cpu['raw_w2'][expert].view(torch.uint8)
    codes=torch.stack((packed&15,packed>>4),-1).reshape(n,k).long()
    expanded=(table[codes]*torch.exp2(scales[expert].float()-127).repeat_interleave(32,-1)).bfloat16()
    assert torch.equal(expanded.view(torch.uint8),logical[expert].view(torch.uint8)),int(expert)
reference=[]
for row in rows:
    result=torch.zeros(n,dtype=torch.float64)
    for slot in range(6):
        expert=int(cpu['topk_ids'][row,slot])
        result+=(logical[expert].double() @ cpu['stage2_input'][row,slot].double())*cpu['topk_weights'][row,slot].double()
    reference.append(result)
reference=torch.stack(reference)
gpu={name:cpu[name].cuda() for name in ('weight13','weight2','stage2_input',
    'sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights')}
for name in ('weight13','weight2'):
    if meta[name]['is_shuffled']:gpu[name].is_shuffled=True
module=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
a=meta['stage2_args'];m=cpu['stage2_input'].shape[0]
out=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
def atomic():
    accum=torch.zeros_like(out,dtype=torch.float32)
    module.ck_moe_stage2(gpu['stage2_input'],gpu['weight13'],gpu['weight2'],gpu['sorted_token_ids'],
        gpu['sorted_expert_ids'],gpu['num_valid_ids'],accum,a['topk'],a['kernel'],None,None,a['block_m'],
        gpu['sorted_weights'],a['quant_type'],a['activation'],a['use_non_temporal_load'])
    out.copy_(accum)
def fixed():
    ck_fixed_slot_stage2(module.ck_moe_stage2,gpu['stage2_input'],gpu['weight13'],gpu['weight2'],
        gpu['sorted_token_ids'],gpu['sorted_expert_ids'],gpu['num_valid_ids'],out,a['topk'],a['kernel'],
        None,None,a['block_m'],gpu['sorted_weights'],a['quant_type'],a['activation'],a['use_non_temporal_load'])
atomic();current=out[rows].cpu();fixed();candidate=out[rows].cpu()
report=dict(real_rows=rows,raw_fp4_selected_experts_exact=len(experts),oracle='FP64 dot on raw-FP4-validated BF16 W2 and frozen BF16 intermediate, weighted FP64 sum',
            row_errors=[],abba=[])
for j,row in enumerate(rows):
    item=dict(row=row)
    for label,value in [('service',cpu['stage2_out'][rows]),('atomic',current),('fixed',candidate)]:
        diff=value[j].double()-reference[j]
        item[label]=dict(max_abs=float(diff.abs().max()),relative_l2=float(diff.norm()/reference[j].norm()),
                        mismatched_to_rounded_fp64=int((value[j]!=reference[j].bfloat16()).sum()))
    report['row_errors'].append(item)
torch.save(dict(rows=rows,reference=reference,atomic=current,fixed=candidate,service=cpu['stage2_out'][rows]),root/'selected-row-reference.pt')
for fn in (atomic,fixed):
    for _ in range(3):fn()
torch.cuda.synchronize()
for arm in ('A','B','B','A'):
    values=[];fn=atomic if arm=='A' else fixed
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record();fn();end.record();end.synchronize();values.append(start.elapsed_time(end))
    report['abba'].append(dict(arm=arm,ms=values,median_ms=statistics.median(values)))
report['atomic_center_ms']=statistics.mean(x['median_ms'] for x in report['abba'] if x['arm']=='A')
report['fixed_center_ms']=statistics.mean(x['median_ms'] for x in report['abba'] if x['arm']=='B')
report['fixed_cost_pct']=100*(report['fixed_center_ms']/report['atomic_center_ms']-1)
(root/'reference.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
