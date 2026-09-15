"""Real fixed-slot Set oracle and complete stage2 ABBA; not service timing."""
import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import statistics

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='6'
import torch
import aiter
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module,ck_fixed_slot_stage2

root=Path(__file__).resolve().parent
build=json.loads(args.build.read_text());assert build['status']=='complete'
module_path=Path(build['module'])
assert hashlib.sha256(module_path.read_bytes()).hexdigest()==build['module_sha256']
spec=importlib.util.spec_from_file_location(build['name'],module_path)
unique=importlib.util.module_from_spec(spec);spec.loader.exec_module(unique)
original=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
mod=fixed_slot_module()
report=dict(diagnostic_only=True,physical_gcd=6,build=build,cases=[],
    baseline_binary_sha256=hashlib.sha256(Path(original.__file__).read_bytes()).hexdigest(),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())

def save():args.output.write_text(json.dumps(report,indent=2)+'\n')

for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
    fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
    cpu={}
    for name in ('weight13','weight2','stage2_input','sorted_token_ids','sorted_expert_ids',
                 'num_valid_ids','sorted_weights','topk_ids','topk_weights'):
        path=fixture/(name+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
        cpu[name]=torch.load(path,map_location='cpu',weights_only=True)
    m,t,k=cpu['stage2_input'].shape;assert t==6 and k==256
    a=meta['stage2_args'];assert a['block_m']==64 and not a['kernel'] and a['quant_type']==0
    valid=int(cpu['num_valid_ids'][0]);encoded=cpu['sorted_token_ids'][:valid].long()
    tokens=encoded&0xffffff;slots=encoded>>24;live=(tokens<m)&(slots<6)
    tokens,slots=tokens[live],slots[live];virtual=tokens*6+slots
    assert len(virtual)==m*6 and torch.equal(torch.sort(virtual).values,torch.arange(m*6))
    assert torch.equal(cpu['sorted_weights'][:valid][live],cpu['topk_weights'][tokens,slots])
    experts=cpu['sorted_expert_ids'].repeat_interleave(64)[:valid][live]
    assert torch.equal(experts,cpu['topk_ids'][tokens,slots])
    data={name:value.cuda() for name,value in cpu.items()}
    for name in ('weight13','weight2'):
        if meta[name]['is_shuffled']:data[name].is_shuffled=True
    inter=data['stage2_input'];ids=data['sorted_token_ids'];weights=data['sorted_weights']
    w1,w2=data['weight13'],data['weight2'];se=data['sorted_expert_ids'];nv=data['num_valid_ids']
    out=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
    def ck(dst,ri=ids,source=inter,topk=6):
        original.ck_moe_stage2(source,w1,w2,ri,se,nv,dst,topk,a['kernel'],None,None,
            64,weights,0,a['activation'],a['use_non_temporal_load'])
    def atomic():
        accum=torch.zeros_like(out,dtype=torch.float32);ck(accum);out.copy_(accum)
    def fixed():
        ck_fixed_slot_stage2(original.ck_moe_stage2,inter,w1,w2,ids,se,nv,out,6,'',None,None,
            64,weights,0,a['activation'],a['use_non_temporal_load'])
    def candidate():
        remapped=torch.empty_like(ids);mod.remap(ids,nv,remapped,m)
        partial=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
        unique.stage2(inter.view(m*6,1,k),w2,remapped,se,nv,weights,partial)
        mod.reduce(partial.view(m,6,4096),out)
    remapped=torch.empty_like(ids);mod.remap(ids,nv,remapped,m)
    partial=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
    partial_ref=torch.empty_like(partial)
    checks=[]
    for mutation in range(5):
        if mutation:
            inter.mul_(.999);weights.mul_(.998)
        partial_ref.zero_();ck(partial_ref,remapped,inter.view(m*6,1,k),1)
        partial.fill_(float('nan'))
        unique.stage2(inter.view(m*6,1,k),w2,remapped,se,nv,weights,partial)
        fixed();expected=out.clone();mod.reduce(partial.view(m,6,4096),out)
        check=dict(mutation=mutation,partial_finite=bool(torch.isfinite(partial).all()),
            partial_byte_exact=torch.equal(partial.view(torch.uint8),partial_ref.view(torch.uint8)),
            output_byte_exact=torch.equal(out.view(torch.uint8),expected.view(torch.uint8)),
            max_abs=float((out.float()-expected.float()).abs().max()))
        checks.append(check)
        assert check['partial_finite'] and check['output_byte_exact'],check
    for replay in range(30):
        partial.fill_(float('nan'));out.fill_(float('nan'))
        unique.stage2(inter.view(m*6,1,k),w2,remapped,se,nv,weights,partial)
        mod.reduce(partial.view(m,6,4096),out)
        assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8)),replay
    del partial,partial_ref,remapped,expected
    inter.copy_(cpu['stage2_input']);weights.copy_(cpu['sorted_weights'])
    samples={name:[] for name in ('atomic','fixed','unique')}
    funcs=dict(atomic=atomic,fixed=fixed,unique=candidate)
    for fn in funcs.values():
        for _ in range(3):fn()
    # Two independent ABBA families, each repeated three cycles.
    families={}
    for control in ('atomic','fixed'):
        records=[]
        for cycle in range(3):
            for arm in (control,'unique','unique',control):
                torch.cuda.synchronize();start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(5):funcs[arm]()
                end.record();end.synchronize();ms=start.elapsed_time(end)/5
                records.append(dict(cycle=cycle,arm=arm,ms=ms))
        centers={arm:statistics.median(r['ms'] for r in records if r['arm']==arm) for arm in (control,'unique')}
        families[control]=dict(records=records,median_ms=centers,speedup=centers[control]/centers['unique'])
    report['cases'].append(dict(fixture=str(fixture),m=m,checks=checks,poisoned_eager_replays=30,
        unique_ownership_checked=True,scratch_bytes=m*6*4096*4,abba=families))
    save();print(m, {key:val['median_ms'] for key,val in families.items()},flush=True)
    del cpu,data,inter,ids,weights,w1,w2,se,nv,out
    torch.cuda.empty_cache()
report['status']='complete';save()
