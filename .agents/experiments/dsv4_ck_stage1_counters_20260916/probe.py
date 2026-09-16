"""Frozen real gate/up stage1: exact replay, uninstrumented timing or counters."""
import argparse
import ctypes
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
from unittest.mock import patch

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--profile',action='store_true')
args=parser.parse_args();assert not args.output.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
root=Path(__file__).resolve().parent
roctx=None
if args.profile:
    roctx=ctypes.CDLL('/opt/rocm/lib/librocprofiler-sdk-roctx.so')
    for name in ('roctxProfilerPause','roctxProfilerResume'):
        fn=getattr(roctx,name);fn.argtypes=[ctypes.c_uint64];fn.restype=ctypes.c_int
    assert roctx.roctxProfilerPause(0)==0
import torch
original_check_output=subprocess.check_output
version_queries=[]
def version_query(command,*a,**kw):
    if isinstance(command,(list,tuple)) and '--version' in command:
        env=dict(kw.get('env') or os.environ)
        removed=[k for k in env if k.startswith('ROCPROF') or k in ('LD_PRELOAD','HSA_TOOLS_LIB')]
        for key in removed:env.pop(key,None)
        kw['env']=env;version_queries.append(dict(command=list(command),removed=removed))
    return original_check_output(command,*a,**kw)
with patch('subprocess.check_output',side_effect=version_query):
    import aiter
assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
fixture=root.parent/'dsv4_c16_real_ck_replay_20260915/capture-v2/fixture'
meta=json.loads((fixture/'manifest.json').read_text())
names=('hidden','weight13','weight2','sorted_token_ids','sorted_expert_ids','num_valid_ids','sorted_weights','stage1_out')
data={}
for name in names:
    path=fixture/(name+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
    data[name]=torch.load(path,map_location='cuda',weights_only=True).contiguous()
module=importlib.import_module('aiter.jit.module_moe_ck2stages_b16_b16_preshuffle_off_b16_dsv4silu_no_mulWeightStage2_')
out=torch.empty_like(data['stage1_out']);assert out.shape==(32767,6,256)
def call():
    module.ck_moe_stage1(data['hidden'],data['weight13'],data['weight2'],data['sorted_token_ids'],
        data['sorted_expert_ids'],data['num_valid_ids'],out,6,meta['stage1_kernel'],None,None,64,
        data['sorted_weights'],0,aiter.ActivationType.Dsv4Silu.value,1,False,None)
for _ in range(3):call()
torch.cuda.synchronize()
exact=torch.equal(out.view(torch.uint8),data['stage1_out'].view(torch.uint8))
assert exact, ('stage1 differs from captured intermediate',float((out.float()-data['stage1_out'].float()).abs().max()))
assert bool(torch.isfinite(out).all())
result=dict(status='running',pci=pci.value.decode(),shape=list(out.shape),stage1_kernel=meta['stage1_kernel'],
    captured_intermediate_byte_exact=exact,profiled=args.profile,
    module=dict(path=module.__file__,sha256=hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()),
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),fixture=str(fixture),
    fixture_manifest_sha256=hashlib.sha256((fixture/'manifest.json').read_bytes()).hexdigest(),version_queries=version_queries)
if args.profile:
    for iteration in range(3):
        torch.cuda.synchronize();assert roctx.roctxProfilerResume(0)==0
        call();torch.cuda.synchronize()
        assert roctx.roctxProfilerPause(0)==0
        assert torch.equal(out.view(torch.uint8),data['stage1_out'].view(torch.uint8))
    result['profiled_exact_replays']=3
else:
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):call()
    for _ in range(100):graph.replay()
    assert torch.equal(out.view(torch.uint8),data['stage1_out'].view(torch.uint8))
    samples=[]
    for _ in range(10):
        begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(5):graph.replay()
        end.record();end.synchronize();samples.append(begin.elapsed_time(end)/5)
    result.update(replay100_exact=True,median_ms=statistics.median(samples),samples_ms=samples)
result['status']='complete'
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
