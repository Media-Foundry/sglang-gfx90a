"""Isolate the unfused first-MHC BF16-Fn shortcut on identical live residuals.

No service mutation. Require both traced services to have stopped. The full-M
input repeats real suffix rows; it is not claimed to reproduce unrelated rows
in the original batch. Compare corresponding rows, order permutations, and an
FP64 mathematical reference. Timings are component-only.
"""
import hashlib
import argparse
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES')=='5'
import torch
from safetensors import safe_open
from sglang.kernels.ops.layernorm import gfx90a_mhc_pre_mix as premix

root=Path(__file__).resolve().parent;repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--integrated',action='store_true')
args=parser.parse_args()
target=root/('first-mhc-integrated.json' if args.integrated else 'first-mhc-oracle.json');assert not target.exists()
torch.set_num_threads(1);torch.set_grad_enabled(False)
assert torch.cuda.device_count()==1 and torch.cuda.mem_get_info()[0]>50*1024**3
for arm in ('mixed','serial'):
    d=root/f'trace-v2-{arm}'
    assert not json.loads((d/f'P32k-tail-policy-trace-v2-{arm}.stop.json').read_text())['remaining']

def sample(arm,stage):
    d=root/f'trace-v2-{arm}'/'trace';stem='layer-0-rank-0-'
    metadata=torch.load(d/(stem+'metadata.pt'),weights_only=False)['metadata']
    assert metadata['prefix_lens'][0]==32768 and metadata['extend_lens'][0]==64
    record=json.loads((d/(stem+stage+'.json')).read_text())
    path=d/record['values_file']
    assert hashlib.sha256(path.read_bytes()).hexdigest()==record['file_sha256']
    raw=torch.load(path,weights_only=False)['samples']
    rows=[metadata['sampled_rows'].index(i) for i in range(64)]
    return raw[rows].contiguous().view(getattr(torch,record['dtype'].removeprefix('torch.'))).cuda()

seed=sample('mixed','attn_residual')
assert torch.equal(seed,sample('serial','attn_residual'))
model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key='layers.0.hc_attn_fn'
with safe_open(str(model/index[key]),framework='pt',device='cpu') as f:
    fn=f.get_tensor(key).float().reshape(24,16384).contiguous().cuda()
digest=lambda x:hashlib.sha256(x.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
record=json.loads((root/'trace-v2-mixed/trace/layer-0-rank-0-hc_attn_fn.json').read_text())
assert digest(fn)==record['sha256']
sources=[Path(__file__),Path(premix.__file__),repo/'python/sglang/kernels/jit/csrc/gemm/gfx90a_mhc_pre_mix.cuh',
         repo/'python/sglang/kernels/ops/layernorm/gfx90a_mhc_prefill_policy.py']
report=dict(status='running',weight_sha256=digest(fn),input_sha256=digest(seed),
    scope=__doc__,integrated=args.integrated,cases=[],sources={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
def save():target.write_text(json.dumps(report,indent=2)+'\n')
def compare(x,y):
    assert x.shape==y.shape and torch.isfinite(x).all() and torch.isfinite(y).all()
    diff=(x.double()-y.double()).abs()
    return dict(byte_exact=torch.equal(x.contiguous().view(torch.uint8),y.contiguous().view(torch.uint8)),
                different_elements=int((x!=y).sum()),max_abs=float(diff.max()),mean_abs=float(diff.mean()))
def call(x,bf16):
    os.environ['SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM']=str(int(bf16))
    if not args.integrated:
        return premix.gfx90a_mhc_pre_mix_wave64(x,fn,1e-6)
    from sglang.srt.layers import dsv4_prefill_experiments as scope
    from sglang.srt.layers import dsv4_prefill_tail_policy as tail
    os.environ['SGLANG_DSV4_PREFILL_MHC_COMMON_UNFUSED']='1'
    os.environ['SGLANG_DSV4_PREFILL_MHC_COMMON_FP32']='1'
    large=scope._mix_pair.set(x.shape[0]>=8192)
    small=tail._active.set(x.shape[0]<8192)
    try:
        return premix.gfx90a_mhc_pre_mix_wave64(x,fn,1e-6)
    finally:
        scope._mix_pair.reset(large);tail._active.reset(small)
save()
try:
    for mutation in range(3):
        small=seed if mutation==0 else (seed.float()*(1-mutation*.013)).bfloat16()
        small=small.contiguous()
        reference=small.flatten(1).double()
        reference=(reference@fn.double().t())*torch.rsqrt(reference.square().mean(1,keepdim=True)+1e-6)
        reference=reference.unsqueeze(1)
        exact_small=call(small,False)
        # M64 currently uses two columns/wave; compare the three-column variant.
        direct=torch.empty_like(exact_small)
        premix._jit_gfx90a_mhc_pre_mix_module().run(small,fn,direct,1e-6)
        entry=dict(mutation=mutation,m64_variants=compare(exact_small,direct),
                   fp32_vs_fp64=compare(exact_small,reference),shapes=[])
        if mutation==0:
            entry['service_small_replay']=compare(exact_small,sample('serial','unfused_mhc_mixes'))
        for m in (65,8192,32576):
            large=small.repeat((m+63)//64,1,1)[:m].contiguous()
            fp32=call(large,False);bf16=call(large,True)
            if args.integrated:
                assert compare(bf16,fp32)['byte_exact'],('integrated contract',mutation,m)
            row_check=compare(fp32[:64],exact_small)
            assert row_check['byte_exact'],(mutation,m,row_check)
            perm=torch.arange(m-1,-1,-1,device='cuda')
            permuted=call(large[perm].contiguous(),False)
            assert compare(permuted[perm],fp32)['byte_exact']
            item=dict(m=m,fp32_shape_exact=row_check,bf16_vs_fp32=compare(bf16[:64],exact_small),
                      bf16_vs_fp64=compare(bf16[:64],reference),permutation_exact=True)
            samples={False:[],True:[]}
            for _ in range(3):
                for flag in (False,True,True,False):
                    start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                    start.record();value=call(large,flag);end.record();end.synchronize()
                    samples[flag].append(start.elapsed_time(end))
            item['median_ms']={'fp32':statistics.median(samples[False]),'bf16_shortcut':statistics.median(samples[True])}
            if mutation==0 and m==32576:
                item['service_large_replay']=compare(bf16[:64],sample('mixed','unfused_mhc_mixes'))
            entry['shapes'].append(item);print('CASE',mutation,m,item,flush=True)
            del large,fp32,bf16,permuted,perm,value
        report['cases'].append(entry);save()
    report['status']='complete';save()
except BaseException:
    report['status']='failed';report['error']=traceback.format_exc();save();raise
