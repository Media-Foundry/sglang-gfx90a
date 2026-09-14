"""Actual MHC dispatcher, scope and allocating wrapper A4/A8 oracle."""
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import torch
import triton
from sglang.kernels.ops.layernorm.mhc import gfx90a_mhc_pre_mix_from_partials_triton
from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse

assert os.environ['HIP_VISIBLE_DEVICES']=='4'
assert os.environ['SGLANG_DSV4_PREFILL_MIX_REUSE4']=='1'
root=Path(__file__).resolve().parent
out=root/'integrated.json';assert not out.exists()
source=root.parent/'dsv4_input_identity_20260914/trace-B1'
fn=torch.load(source/'layer_0_rank_0_hc_ffn_fn.pt',map_location='cuda',weights_only=True).contiguous()
seed=torch.load(source/'layer_0_rank_0_ffn_mhc_residual.pt',map_location='cuda',weights_only=True)
torch.manual_seed(20260915)
result=dict(status='running',scope=__doc__,results=[],source_sha256={})
for name in ('mhc.py','gfx90a_mhc_premix_reuse.py','gfx90a_mhc_premix_reuse8.py'):
    path=root.parents[2]/'python/sglang/kernels/ops/layernorm'/name
    result['source_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()

for m in (8192,32765,32768,65536):
    x=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    rms=x.float().square().view(m,64,256).sum(-1)
    def call(group):
        os.environ['SGLANG_DSV4_PREFILL_MIX_GROUP_SIZE']=str(group)
        token=_mix_reuse.set(True)
        try:return gfx90a_mhc_pre_mix_from_partials_triton(x,fn,rms,1e-6)
        finally:_mix_reuse.reset(token)
    for _ in range(3):call(4);call(8)
    samples={4:[],8:[]}
    for _ in range(3):
        for group in (4,8,8,4):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(5):call(group)
            end.record();end.synchronize();samples[group].append(start.elapsed_time(end)/5)
    for _ in range(10):
        x.normal_();fn.normal_(std=.01)
        rms.copy_(x.float().square().view(m,64,256).sum(-1))
        a,b=call(4),call(8)
        assert torch.equal(a.view(torch.int32),b.view(torch.int32))
        assert not _mix_reuse.get()
    item=dict(m=m,mutations=10,bits_exact=True,samples_ms=samples,
        median_ms={k:median(v) for k,v in samples.items()})
    result['results'].append(item)
    out.write_text(json.dumps(result,indent=2)+'\n');print(item,flush=True)
    del x,rms,a,b
    torch.cuda.empty_cache()
result['status']='complete';out.write_text(json.dumps(result,indent=2)+'\n')
