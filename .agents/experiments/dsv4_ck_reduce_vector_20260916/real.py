"""Real CK stage2 -> scalar/vec4 reduction ABBA; includes remap and allocations."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
assert os.environ.get('HIP_VISIBLE_DEVICES')=='6'
import torch
from sglang.kernels.jit.utils import load_jit
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module
root=Path(__file__).resolve().parent
target=root/'real.json';assert not target.exists()
screen=json.loads((root/'screen.json').read_text());assert screen['status']=='complete'
mod=load_jit('dsv4_reduce_vec4_screen',cuda_files=[str(root/'vector.cuh')],
    cuda_wrappers=[('bf16','sglang::ReduceVectorScreen::bf16'),('fp32','sglang::ReduceVectorScreen::fp32')],
    extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
    extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
refmod=fixed_slot_module()
build_path=root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json'
build=json.loads(build_path.read_text());assert build['status']=='complete'
assert hashlib.sha256(Path(build['module']).read_bytes()).hexdigest()==build['module_sha256']
spec=importlib.util.spec_from_file_location(build['name'],build['module'])
unique=importlib.util.module_from_spec(spec);spec.loader.exec_module(unique)
report=dict(status='running',scope='Historical real inputs; full unique CK stage2 incl remap/allocation/reducer, not whole routed MoE or E2E',
    module_sha256=build['module_sha256'],physical_gcd=6,cases=[])
def save():target.write_text(json.dumps(report,indent=2)+'\n')
save()
for location in ('dsv4_cached_ck_drift_20260915/capture','dsv4_c16_real_ck_replay_20260915/capture-v2'):
    fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
    data={}
    for name in ('weight2','stage2_input','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights'):
        path=fixture/(name+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
        data[name]=torch.load(path,map_location='cpu',weights_only=True).cuda()
    inter=data['stage2_input'];m,t,k=inter.shape;assert t==6 and k==256
    out=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
    remapped=torch.empty_like(data['sorted_token_ids'])
    refmod.remap(data['sorted_token_ids'],data['num_valid_ids'],remapped,m)
    partial=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
    checks=[]
    for mutation in range(10):
        if mutation:inter.mul_(.999);data['sorted_weights'].mul_(.997)
        partial.fill_(float('nan'))
        unique.stage2(inter.view(m*6,1,k),data['weight2'],remapped,data['sorted_expert_ids'],data['num_valid_ids'],data['sorted_weights'],partial)
        assert bool(torch.isfinite(partial).all())
        for dtype in (torch.bfloat16,torch.float32):
            ref=torch.empty((m,4096),device='cuda',dtype=dtype);candidate=torch.empty_like(ref)
            refcall=refmod.reduce if dtype==torch.bfloat16 else refmod.reduce_float
            candcall=mod.bf16 if dtype==torch.bfloat16 else mod.fp32
            refcall(partial.view(m,6,4096),ref);candcall(partial.view(m,6,4096),candidate,1664)
            exact=torch.equal(ref.view(torch.uint8),candidate.view(torch.uint8))
            checks.append(dict(mutation=mutation,dtype=str(dtype),byte_exact=exact));assert exact
            del ref,candidate
    del partial,remapped
    def call(vector):
        ids=torch.empty_like(data['sorted_token_ids'])
        refmod.remap(data['sorted_token_ids'],data['num_valid_ids'],ids,m)
        p=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
        unique.stage2(inter.view(m*6,1,k),data['weight2'],ids,data['sorted_expert_ids'],data['num_valid_ids'],data['sorted_weights'],p)
        if vector:mod.bf16(p.view(m,6,4096),out,1664)
        else:refmod.reduce(p.view(m,6,4096),out)
    for _ in range(3):call(False);call(True)
    records=[]
    for cycle in range(5):
        for vector in (False,True,True,False):
            torch.cuda.synchronize();a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            a.record()
            for _ in range(5):call(vector)
            b.record();b.synchronize()
            records.append(dict(cycle=cycle,arm='vec4' if vector else 'scalar',ms=a.elapsed_time(b)/5))
    med={arm:statistics.median(r['ms'] for r in records if r['arm']==arm) for arm in ('scalar','vec4')}
    report['cases'].append(dict(m=m,fixture=str(fixture),checks=checks,records=records,median_ms=med,speedup=med['scalar']/med['vec4']))
    save();print('REAL',m,med,flush=True)
    del inter,out,data
    torch.cuda.empty_cache()
report['status']='complete';save()
