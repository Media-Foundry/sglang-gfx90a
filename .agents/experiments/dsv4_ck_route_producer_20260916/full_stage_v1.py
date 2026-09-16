"""Complete current routed helper versus explicit route-major producer.

Includes raw FP4/scale conversion, weight workspace, sorting, allocations,
both CK stages and final fixed reduction. No service or1M-pool claim.
"""
import ctypes
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import statistics
import traceback

root=Path(__file__).resolve().parent
assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
settings={
    'SGLANG_DSV4_GFX90A_BF16_CK_STAGE2_FP32':'1',
    'SGLANG_DSV4_GFX90A_BF16_CK_BLOCK64_V1':'1',
    'SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT':'1',
    'SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST':str(root.parent/'dsv4_prefill_extras_build_20260916/build-v1/unique/manifest.json'),
    'SGLANG_DSV4_DEBUG_CK_REDUCE_VEC4':'1',
    'SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT':'1',
    'SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT_CHECK':'0',
    'AITER_DSV4_DEBUG_KEEP_BF16_WEIGHTS':'0',
    'AITER_DSV4_DEBUG_SHUFFLE_BF16_WEIGHTS':'1',
    'AITER_DSV4_DEBUG_RAW_LOGICAL_B_STAGE1':'0',
    'SGLANG_DSV4_DEBUG_CK_STAGE_CAPTURE_DIR':'',
    'AITER_DSV4_DEBUG_ACTIVATION':'',
    'SGLANG_DSV4_GFX90A_BF16_CK_BLOCK_M':'64',
    'SGLANG_DSV4_GFX90A_BF16_CK_STAGE1_KERNEL':'',
    'SGLANG_DSV4_GFX90A_BF16_CK_STAGE2_KERNEL':'',
}
os.environ.update(settings)
import torch
from runner import RouteProducer
from sglang.kernels.jit.utils import load_jit
import sglang.kernels.ops.moe.gfx90a_bf16_batched_moe as current
from sglang.kernels.ops.moe.gfx90a_bf16_direct_rows import run as dequant
from sglang.srt.layers.dsv4_prefill_experiments import _mix_pair

target=root/'full-stage-v1.json';assert not target.exists()
torch.set_grad_enabled(False);torch.set_num_threads(1)
assert torch.cuda.mem_get_info()[0]>50*1024**3
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),settings=settings,cases=[],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in
             [Path(__file__),root/'runner.py',root/'metadata.cuh',Path(current.__file__)]})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def load(manifest):
    record=json.loads(Path(manifest).read_text());assert record['status']=='complete'
    assert hashlib.sha256(Path(record['module']).read_bytes()).hexdigest()==record['module_sha256']
    for path,digest in record['sources'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==digest,path
    spec=importlib.util.spec_from_file_location(record['name'],record['module'])
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module
save()
scope=_mix_pair.set(True)  # Standalone scope simulation, not a service-dispatch test.
try:
    route=load(root/'build-v1/route/manifest.json')
    ck=load(settings['SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST'])
    maps=load_jit('dsv4_route_producer_metadata',cuda_files=[str(root/'metadata.cuh')],
        cuda_wrappers=[('metadata','sglang::RouteMetadata::run'),('reduce','sglang::RouteMajor::bf16')],
        extra_include_paths=[str(root.parents[2]/'python/sglang/kernels/jit/csrc')],
        extra_cuda_cflags=['-O3','-fno-fast-math','-ffp-contract=off'])
    compute=RouteProducer(route,ck,maps)
    for m in (8192,16384,32767):
        location=('dsv4_cached_ck_drift_20260915/capture' if m==8192
                  else 'dsv4_c16_real_ck_replay_20260915/capture-v2')
        fixture=root.parent/location/'fixture';meta=json.loads((fixture/'manifest.json').read_text())
        cpu={}
        for name in ('hidden','topk_ids','topk_weights','raw_w13','raw_s13','raw_w2','raw_s2','weight13','weight2'):
            path=fixture/(name+'.pt');assert hashlib.sha256(path.read_bytes()).hexdigest()==meta[name]['file_sha256']
            value=torch.load(path,map_location='cpu',weights_only=True)
            if name in ('hidden','topk_ids','topk_weights'):value=value[:m]
            cpu[name]=value.contiguous()
        data={k:v.cuda() for k,v in cpu.items()}
        hidden=data['hidden'];ids=data['topk_ids'];weights=data['topk_weights']
        raw13=data['raw_w13'];s13=data['raw_s13'];raw2=data['raw_w2'];s2=data['raw_s2']
        shuffled=meta['scales_shuffled'];assert isinstance(shuffled,bool)
        def baseline():
            return current.gfx90a_bf16_ck_moe(hidden,ids,weights,raw13,s13,raw2,s2,
                                             scales_shuffled=shuffled)
        # Allocate the SAME shared per-layer weight workspace via production.
        expected=baseline();torch.cuda.synchronize()
        w13,w2=current._ck_weight_workspaces[torch.cuda.current_device()]
        assert torch.equal(w13.view(torch.int16),data['weight13'].view(torch.int16)),('expanded gate',m)
        assert torch.equal(w2.view(torch.int16),data['weight2'].view(torch.int16)),('expanded down',m)
        fallback13=current._jit_dequant(256,512,4096,1664).run_shuffled
        fallback2=current._jit_dequant(256,4096,256,1664).run_shuffled
        def candidate():
            a=current._logical_a16w4_scales(s13,256,512,128,gate_up=True) if shuffled else s13
            b=current._logical_a16w4_scales(s2,256,4096,8,gate_up=False) if shuffled else s2
            dequant('gate',raw13.view(torch.uint8),a.view(torch.uint8).reshape(256,512,128),w13,fallback13)
            dequant('down',raw2.view(torch.uint8),b.view(torch.uint8).reshape(256,4096,8),w2,fallback2)
            return compute(hidden,ids,weights,w13,w2)
        checks=[]
        for mutation in range(4):
            if mutation==1:hidden.mul_(.999);weights.mul_(.997)
            if mutation==2:ids.copy_((ids+137)%256)
            if mutation==3:
                hidden.copy_(hidden.flip(0));ids.copy_(ids.flip(0));weights.copy_(weights.flip(0))
            expected=baseline();got=candidate()
            assert bool(torch.isfinite(expected).all()) and bool(torch.isfinite(got).all())
            assert torch.equal(expected.view(torch.int16),got.view(torch.int16)),('full routed',m,mutation)
            checks.append(dict(mutation=mutation,output_byte_exact=True))
        for name in ('hidden','topk_ids','topk_weights'):data[name].copy_(cpu[name])
        for _ in range(3):baseline();candidate()
        samples=[]
        for cycle in range(3):
            for arm in ('A','B','B','A'):
                torch.cuda.synchronize();begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                call=baseline if arm=='A' else candidate
                begin.record()
                for _ in range(3):result=call()
                end.record();end.synchronize()
                samples.append(dict(cycle=cycle,arm=arm,ms=begin.elapsed_time(end)/3))
        med={arm:statistics.median(s['ms'] for s in samples if s['arm']==arm) for arm in ('A','B')}
        memory={}
        del expected,got,result
        for arm,call in [('A',baseline),('B',candidate)]:
            torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            initial=torch.cuda.memory_allocated();result=call();torch.cuda.synchronize()
            memory[arm]=dict(initial_allocated=initial,peak_allocated=torch.cuda.max_memory_allocated(),
                             incremental_peak=torch.cuda.max_memory_allocated()-initial)
            del result
        report['cases'].append(dict(m=m,fixture=str(fixture),scales_shuffled=shuffled,
            checks=checks,expanded_weights_exact=True,median_ms=med,speedup=med['A']/med['B'],
            samples=samples,memory=memory))
        save();print('FULL ROUTED',m,med,'speedup',med['A']/med['B'],flush=True)
        del cpu,data,hidden,ids,weights,raw13,s13,raw2,s2,w13,w2
        torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
finally:
    _mix_pair.reset(scope)
