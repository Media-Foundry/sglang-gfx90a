"""Full MHC boundary component, not model-quality or service acceptance.

Repeated real tensor seeds with real checkpoint Fn/base/scale/norm. They are
not asserted to be matching causal inputs for each sampled layer. Compare
identical inputs under dispatch hints; do not modify checkpoint tensors.
"""
import argparse
import contextlib
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
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
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,default=root/'oracle.json')
target=parser.parse_args().output
assert not target.exists()
# Only allocation wrappers are replaced for this single-GCD/no-distributed
# component test. No math kernel is replaced; TP pre-mix owner stays off.
mhc.get_tp_group=lambda:None
mhc.is_allocation_symmetric=lambda:False
mhc.use_symmetric_memory=lambda *a,**kw:contextlib.nullcontext()
tokens=[(v,v.set(True)) for v in (scope._post_reuse,scope._mix_reuse,scope._mix_pair)]
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
    with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:
        return f.get_tensor(key).cuda()
def equal(a,b):return all(torch.equal(x.view(torch.uint8),y.view(torch.uint8)) for x,y in zip(a,b,strict=True))
torch.manual_seed(20260916)
try:
    for layer in (0,20,42):
        fn=weight(f'layers.{layer}.hc_ffn_fn').float().reshape(24,16384).contiguous()
        fn16=fn.half()
        scale=weight(f'layers.{layer}.hc_ffn_scale').float().reshape(3).contiguous()
        base=weight(f'layers.{layer}.hc_ffn_base').float().reshape(24).contiguous()
        norm=weight(f'layers.{layer}.ffn_norm.weight').bfloat16().contiguous()
        for m in ((1,17,8191,8192,8193,32767,32768,65536) if layer==0 else (8192,32767)):
            x,residual,post,comb=[s.repeat(triton.cdiv(m,len(s)),*([1]*(s.ndim-1)))[:m].contiguous() for s in seed]
            post=post.view(m,4);comb=comb.view(m,4,4)
            def call(hint,common):
                os.environ['SGLANG_DSV4_PREFILL_MHC_COMMON_FP32']=str(int(common))
                return mhc.mhc_fused_post_pre(x,residual,post,comb,fn,scale,base,
                    1.e-6,1.e-6,1.e-6,2.,20,norm_weight=norm,norm_eps=1.e-6,
                    global_batch_size=hint,fn_fp16=fn16)
            checks=[]
            for mutation in range(3):
                if mutation:
                    x.mul_(.999);residual.mul_(1.001);post.mul_(.999);comb.mul_(1.001)
                ref=call(2 if m>=8192 else 1,False)
                candidate=call(1,True)
                assert equal(ref,candidate),(layer,m,mutation)
                assert all(bool(torch.isfinite(t).all()) for t in candidate)
                checks.append(dict(mutation=mutation,all_four_byte_exact=True))
            # Reverse rows while preserving shape; compare reference/candidate
            # on that same permutation, not against legacy singleton math.
            x,residual,post,comb=[t.flip(0).contiguous() for t in (x,residual,post,comb)]
            ref=call(2 if m>=8192 else 1,False);candidate=call(1,True)
            assert equal(ref,candidate)
            legacy=call(1,False)
            delta=[float((a.float()-b.float()).abs().max()) for a,b in zip(legacy,candidate)]
            samples={name:[] for name in ('legacy','common')}
            if m>=8192:
                for _ in range(3):
                    for name,enabled in [('legacy',False),('common',True),('common',True),('legacy',False)]:
                        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                        start.record();out=call(1,enabled);end.record();end.synchronize()
                        samples[name].append(start.elapsed_time(end))
                del out
            record=dict(layer=layer,m=m,checks=checks,reversed_rows_exact=True,
                legacy_max_abs=delta,samples_ms=samples,
                median_ms={name:statistics.median(v) for name,v in samples.items() if v})
            report['cases'].append(record);save();print('PASS',layer,m,record['median_ms'],delta,flush=True)
            del ref,candidate,legacy,x,residual,post,comb
            torch.cuda.empty_cache()
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==h for p,h in report['sources'].items())
    report['status']='complete';save()
except Exception as error:
    report['status']='failed';report['error']=repr(error);save()
    raise
finally:
    for v,t in reversed(tokens):v.reset(t)
