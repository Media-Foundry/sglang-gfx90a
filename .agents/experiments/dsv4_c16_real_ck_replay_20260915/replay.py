"""Frozen real layer21/rank5 CK replay on one otherwise idle GCD. Not E2E."""
import hashlib
import importlib
import json
import os
from pathlib import Path

import torch

root=Path(__file__).resolve().parent
fixture=root/'capture-v2/fixture'
manifest=json.loads((fixture/'manifest.json').read_text())
assert (root/'capture-v2/complete.json').exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5', 'Reserve physical GCD5 only'
for key,value in manifest['provenance']['environment'].items():
    if 'DEBUG_CK_STAGE_CAPTURE' not in key and 'DEBUG_FIRST_DIV' not in key:
        os.environ[key]=value
os.environ.pop('SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR',None)

import aiter
from sglang.kernels.ops.debug import dsv4_ck_stage_capture as capture
from sglang.kernels.ops.moe import gfx90a_bf16_batched_moe as helper
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import ck_fixed_slot_stage2

helper._capture_current=capture.current  # Offline callback only, never service configuration.
torch.cuda.set_device(0)
data={}
for name,record in manifest.items():
    if not isinstance(record,dict) or 'file_sha256' not in record:continue
    path=fixture/(name+'.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['file_sha256'],name
    cpu=torch.load(path,weights_only=True,map_location='cpu')
    assert hashlib.sha256(cpu.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()==record['sha256']
    if name in ('input_ids','positions'):continue
    tensor=cpu.cuda()
    if record['is_shuffled']:tensor.is_shuffled=True
    data[name]=tensor
assert torch.equal(data['stage1_out'].view(torch.uint8),data['stage2_input'].view(torch.uint8))


def delta(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    bits=(a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8))
    changed=bits.reshape(a.shape[0],-1).any(1).nonzero().flatten()
    count=int(bits.sum())
    result=dict(changed_bytes=count,changed_rows=len(changed),first_rows=changed[:32].cpu().tolist())
    if a.dtype.is_floating_point:
        x,y=a.float(),b.float()
        result.update(max_abs=float((x-y).abs().max()),relative_l2=float(
            torch.linalg.vector_norm(x-y)/torch.linalg.vector_norm(x).clamp_min(1e-30)),
            finite=bool(torch.isfinite(x).all() and torch.isfinite(y).all()))
    return result


keep={'stage1_out','stage2_input','stage2_accum','stage2_out','sorted_token_ids',
      'sorted_expert_ids','num_valid_ids','sorted_weights'}
class Observer:
    def __init__(self):self.values={};self.info_values={}
    def tensor(self,name,value):
        if name in keep:self.values[name]=value.detach().clone()
        elif name in ('weight13','weight2'):
            assert torch.equal(value.view(torch.uint8),data[name].view(torch.uint8)),name
    def info(self,name,value):self.info_values[name]=value


report=dict(diagnostic_only=True,physical_gcd=5,real_capture=manifest['provenance'],full=[],atomic=[],fixed=[])
valid=int(data['num_valid_ids'][0]);block=manifest['stage2_args']['block_m']
def meaningful(name,x):
    if name in ('sorted_token_ids','sorted_weights'):return x[:valid]
    if name=='sorted_expert_ids':return x[:(valid+block-1)//block]
    return x

# Full pipeline resubmits dequant + sorter + stage1 + stage2 with fixed inputs.
first=None
for iteration in range(3):
    observer=Observer();token=capture._active.set(observer)
    try:
        output=helper.gfx90a_bf16_ck_moe(data['hidden'],data['topk_ids'],data['topk_weights'],
            data['raw_w13'],data['raw_s13'],data['raw_w2'],data['raw_s2'],
            scales_shuffled=manifest['scales_shuffled'])
    finally:capture._active.reset(token)
    torch.cuda.synchronize()
    assert observer.info_values['stage2_args']==manifest['stage2_args']
    record=dict(iteration=iteration,versus_service={},versus_first={})
    for name,x in observer.values.items():
        record['versus_service'][name]=delta(meaningful(name,x),meaningful(name,data[name]))
        if first is not None:record['versus_first'][name]=delta(meaningful(name,x),meaningful(name,first[name]))
    if first is None:first=observer.values
    report['full'].append(record)
    print('FULL',iteration,{k:v['changed_bytes'] for k,v in record['versus_service'].items()},flush=True)
del first,observer,output

module=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
args=manifest['stage2_args']
report['module']=dict(path=module.__file__,sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest())
def launch(out,inter=None,ids=None,topk=None):
    module.ck_moe_stage2(data['stage2_input'] if inter is None else inter,
        data['weight13'],data['weight2'],data['sorted_token_ids'] if ids is None else ids,
        data['sorted_expert_ids'],data['num_valid_ids'],out,args['topk'] if topk is None else topk,
        args['kernel'],None,None,args['block_m'],data.get('sorted_weights'),
        args['quant_type'],args['activation'],args['use_non_temporal_load'])

accum=torch.empty_like(data['stage2_accum']);bf16=torch.empty_like(data['stage2_out'])
reference=None
for iteration in range(30):
    accum.zero_();launch(accum);bf16.copy_(accum)
    if reference is None:reference=(accum.clone(),bf16.clone())
    record=dict(iteration=iteration,fp32=delta(accum,reference[0]),bf16=delta(bf16,reference[1]))
    if iteration==0:record['versus_service']=delta(bf16,data['stage2_out'])
    if record['bf16']['changed_rows']:
        rows=(bf16.view(torch.uint8)!=reference[1].view(torch.uint8)).any(1).nonzero().flatten()[:32]
        path=root/f'atomic-changed-{iteration}.pt'
        torch.save(dict(rows=rows.cpu(),a=reference[1][rows].cpu(),b=bf16[rows].cpu()),path)
        record['changed_values']=dict(file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    report['atomic'].append(record)
    print('ATOMIC',iteration,record['fp32']['changed_bytes'],record['bf16']['changed_bytes'],flush=True)
del reference
reference=None
for iteration in range(30):
    ck_fixed_slot_stage2(module.ck_moe_stage2,data['stage2_input'],data['weight13'],data['weight2'],
        data['sorted_token_ids'],data['sorted_expert_ids'],data['num_valid_ids'],bf16,args['topk'],
        args['kernel'],None,None,args['block_m'],data.get('sorted_weights'),args['quant_type'],
        args['activation'],args['use_non_temporal_load'])
    if reference is None:reference=bf16.clone()
    record=dict(iteration=iteration,bf16=delta(bf16,reference))
    if iteration==0:record['versus_service']=delta(bf16,data['stage2_out'])
    report['fixed'].append(record)
    print('FIXED',iteration,record['bf16']['changed_bytes'],flush=True)
(root/'replay.json').write_text(json.dumps(report,indent=2)+'\n')
print('COMPLETE',flush=True)
