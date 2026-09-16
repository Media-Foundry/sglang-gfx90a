"""MHC small-tail arithmetic screen using real tensor seeds/checkpoint weights.

No production changes. Seeds are not claimed to be causal inputs of every
sampled layer. Compare legacy singleton,explicit FP32/20 and repeated large-M
rows. This isolates a boundary contract,not the whole-model drift root cause.
"""
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
# Same effective single-request MHC knobs as the measured large-prefill profile.
flags={
    'SGLANG_DSV4_PREFILL_MHC_COMMON_FP32':'1',
    'SGLANG_DSV4_PREFILL_POST_FUSED4':'1','SGLANG_DSV4_PREFILL_MIX_REUSE4':'1',
    'SGLANG_DSV4_PREFILL_MIX_PAIR_COLUMNS':'1','SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE':'8',
    'SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE':'1','SGLANG_DSV4_DEBUG_PREFILL_MIX_OWNER':'0',
    'SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA':'0','SGLANG_DSV4_GFX90A_BF16_MHC_DOT':'0',
    'SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS':'0','SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20':'0',
    'SGLANG_DSV4_GFX90A_FUSE_MHC_POST_RMS_PARTIALS':'1',
    'SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL':'1',
    'SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX':'1',
    'SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS':'1',
    'SGLANG_DSV4_GFX90A_FP16_MHC_DOT':'1','SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS':'8',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_SINKHORN':'1',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL':'0',
    'SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE':'0',
    'SGLANG_DSV4_GFX90A_MHC_BLOCK_K':'1024',
}
os.environ.update(flags)
import torch
import triton
from safetensors import safe_open
from sglang.kernels.ops.layernorm import mhc
from sglang.srt.layers import dsv4_prefill_experiments as scope

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'oracle-v1.json';assert not target.exists()
torch.set_grad_enabled(False);torch.set_num_threads(1)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
mhc.get_tp_group=lambda:None
mhc.is_allocation_symmetric=lambda:False
mhc.use_symmetric_memory=lambda *a,**kw:contextlib.nullcontext()
hip=ctypes.CDLL('/opt/rocm/lib/libamdhip64.so');pci=ctypes.create_string_buffer(32)
assert hip.hipDeviceGetPCIBusId(pci,32,0)==0
fixture=root.parent/'dsv4_input_identity_20260914/trace-B1'
paths=[fixture/f'layer_0_rank_0_{n}.pt' for n in ('attn_out','ffn_mhc_residual','ffn_mhc_post','ffn_mhc_comb')]
seed=[torch.load(p,map_location='cuda',weights_only=True) for p in paths]
model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
source_paths=[Path(__file__),repo/'python/sglang/kernels/ops/layernorm/mhc.py',
    repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_prefill_policy.py',*paths]
report=dict(status='running',scope=__doc__,pci=pci.value.decode(),flags=flags,cases=[],
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def weight(key):
    with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:return f.get_tensor(key).cuda()
def compare(a,b):
    result=[]
    for name,x,y in zip(('residual','post','comb','normalized'),a,b,strict=True):
        assert x.shape==y.shape and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())
        result.append(dict(name=name,byte_exact=torch.equal(x.view(torch.uint8),y.view(torch.uint8)),
            max_abs=float((x.float()-y.float()).abs().max()),unequal=int((x!=y).sum())))
    return result
def repeat(s,m):return s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous()
save()
try:
    for layer in (0,20,42):
        fn=weight(f'layers.{layer}.hc_ffn_fn').float().reshape(24,16384).contiguous();fn16=fn.half()
        scale=weight(f'layers.{layer}.hc_ffn_scale').float().reshape(3).contiguous()
        base=weight(f'layers.{layer}.hc_ffn_base').float().reshape(24).contiguous()
        norm=weight(f'layers.{layer}.ffn_norm.weight').bfloat16().contiguous()
        for m in ((64,256,2048,2560,8191,8192) if layer==0 else (64,256,2560)):
            data=[repeat(t,m) for t in seed];data[2]=data[2].view(m,4);data[3]=data[3].view(m,4,4)
            def call(values,hint,large_scope=False,trace=False):
                tokens=[(v,v.set(large_scope)) for v in (scope._post_reuse,scope._mix_reuse,scope._mix_pair)]
                seen=[]
                def profiler(frame,event,arg):
                    if event=='call' and frame.f_code.co_filename==mhc.__file__:
                        seen.append(frame.f_code.co_name)
                if trace:sys.setprofile(profiler)
                try:
                    result=mhc.mhc_fused_post_pre(*values,fn,scale,base,1e-6,1e-6,1e-6,2.,20,
                        norm_weight=norm,norm_eps=1e-6,global_batch_size=hint,fn_fp16=fn16)
                finally:
                    if trace:sys.setprofile(None)
                    for v,t in reversed(tokens):v.reset(t)
                return result,sorted(set(seen))
            legacy,legacy_paths=call(data,1,trace=True)
            common,common_paths=call(data,None,trace=True)
            batch2,_=call(data,2)
            assert all(q['byte_exact'] for q in compare(common,batch2))
            large_data=[repeat(t,8192) for t in data]
            large,_=call(large_data,1,large_scope=True)
            large_prefix=[t[:m] for t in large]
            checks=[]
            for mutation in range(3):
                if mutation:
                    for t in data:t.mul_(.999)
                a,_=call(data,1);b,_=call(data,None)
                checks.append(dict(mutation=mutation,legacy_vs_common=compare(a,b)))
            # Timing without Python profiling, no large-M/small-M E2E inference.
            samples={'legacy':[],'common':[]}
            for cycle in range(3):
                for name,hint in (('legacy',1),('common',None),('common',None),('legacy',1)):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();result,_=call(data,hint);end.record();end.synchronize()
                    samples[name].append(start.elapsed_time(end));del result
            item=dict(layer=layer,m=m,legacy_paths=legacy_paths,common_paths=common_paths,
                initial_legacy_vs_common=compare(legacy,common),common_small_vs_large=compare(common,large_prefix),
                batch2_common_exact=True,checks=checks,samples_ms=samples,
                median_ms={k:statistics.median(v) for k,v in samples.items()})
            report['cases'].append(item);save();print('CASE',layer,m,item['median_ms'],item['initial_legacy_vs_common'],flush=True)
            del data,legacy,common,batch2,large_data,large,large_prefix,a,b
            torch.cuda.empty_cache()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
