"""Isolated real W2 fixture: collect stage2/reducer counters, not E2E timing."""
import argparse
import ctypes
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
from unittest.mock import patch

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--profile',action='store_true')
args=p.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='6'
root=Path(__file__).resolve().parent
roctx=None
if args.profile:
    roctx=ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
    for name in ('roctxProfilerPause','roctxProfilerResume'):
        fn=getattr(roctx,name);fn.argtypes=[ctypes.c_uint64];fn.restype=ctypes.c_int
    assert roctx.roctxProfilerPause(0)==0
import torch
version_queries=[]
original_check_output=subprocess.check_output
def version_query(command,*args,**kwargs):
    if isinstance(command,(list,tuple)) and '--version' in command:
        env=dict(kwargs.get('env') or os.environ)
        removed=[k for k in env if k.startswith('ROCPROF') or k in ('LD_PRELOAD','HSA_TOOLS_LIB')]
        for key in removed:env.pop(key,None)
        kwargs['env']=env
        version_queries.append(dict(command=list(command),removed_profiler_keys=removed))
    return original_check_output(command,*args,**kwargs)
# Only read-only child version probes are uninstrumented; parent profiler state
# and every GPU dispatch remain untouched. Never synthesize version output.
with patch('subprocess.check_output',side_effect=version_query):
    import aiter
assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
from sglang.kernels.ops.moe.gfx90a_ck_fixed_slot import fixed_slot_module

fixture=root.parent/'dsv4_c16_real_ck_replay_20260915/capture-v2/fixture'
meta=json.loads((fixture/'manifest.json').read_text())
names=('weight13','weight2','stage2_input','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights')
cpu={}
for name in names:
    path=fixture/(name+'.pt')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
    cpu[name]=torch.load(path,map_location='cpu',weights_only=True)
data={name:value.cuda() for name,value in cpu.items()}
inter=data['stage2_input'];m,t,k=inter.shape;assert (m,t,k)==(32767,6,256)
manifest_path=root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json'
manifest=json.loads(manifest_path.read_text());assert manifest['status']=='complete'
assert hashlib.sha256(Path(manifest['module']).read_bytes()).hexdigest()==manifest['module_sha256']
spec=importlib.util.spec_from_file_location(manifest['name'],manifest['module'])
unique=importlib.util.module_from_spec(spec);spec.loader.exec_module(unique)
original=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_f32_silu_no_mulWeightStage2_')
mod=fixed_slot_module();remapped=torch.empty_like(data['sorted_token_ids'])
mod.remap(data['sorted_token_ids'],data['num_valid_ids'],remapped,m)
partial=torch.empty((m*6,4096),device='cuda',dtype=torch.float32)
reference=torch.zeros_like(partial);a=meta['stage2_args']
original.ck_moe_stage2(inter.view(m*6,1,k),data['weight13'],data['weight2'],remapped,
    data['sorted_expert_ids'],data['num_valid_ids'],reference,1,'',None,None,64,
    data['sorted_weights'],0,a['activation'],a['use_non_temporal_load'])
out=torch.empty((m,4096),device='cuda',dtype=torch.bfloat16)
def call():
    unique.stage2(inter.view(m*6,1,k),data['weight2'],remapped,data['sorted_expert_ids'],
        data['num_valid_ids'],data['sorted_weights'],partial)
    mod.reduce(partial.view(m,6,4096),out)
for _ in range(3):call()
torch.cuda.synchronize()
assert torch.equal(partial.view(torch.uint8),reference.view(torch.uint8))
expected=out.clone()
for iteration in range(3):
    torch.cuda.synchronize()
    if roctx:assert roctx.roctxProfilerResume(0)==0
    call();torch.cuda.synchronize()
    if roctx:assert roctx.roctxProfilerPause(0)==0
    assert torch.equal(out.view(torch.uint8),expected.view(torch.uint8))
result=dict(status='complete',physical_gcd=6,fixture=str(fixture),
    fixture_scope='Real original-V4 historical routed inputs, not a new corrected-sink service capture',
    shape=[m,t,k],partial_byte_exact=True,repeated_output_exact=3,profiled=args.profile,
    scope='Only unique stage2 and fixed reducer; profiler durations are not uninstrumented service timing',
    version_queries=version_queries,
    module_sha256=manifest['module_sha256'],script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
