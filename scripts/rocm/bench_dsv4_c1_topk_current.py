#!/usr/bin/env python3
"""C1 installed fallback versus legacy HIP sqrtsoftplus, isolated only."""
import json
import statistics
from pathlib import Path

import torch
import aiter
from safetensors import safe_open
from sglang.kernels.ops.moe.gfx90a_grouped_router import gfx90a_sqrtsoftplus_router
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.srt.environ import envs


def main():
    assert not envs.SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER.get()
    assert not envs.SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER.get()
    assert envs.SGLANG_DSV4_GFX90A_ROUTER_NUM_WARPS.get() == 1
    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
    root=Path('/home/pc/models/modelscope')
    index=json.loads((root/'model.safetensors.index.json').read_text())['weight_map']
    key='layers.20.ffn.gate.bias'
    with safe_open(root/index[key],framework='pt',device='cpu') as f:
        bias=f.get_tensor(key).to(torch.bfloat16).cuda()
    torch.manual_seed(20908)
    scores=torch.randn((1,256),device='cuda',dtype=torch.bfloat16)
    outputs=[]; graphs=[]
    def current():
        assert not hasattr(aiter,'topk_gating'), 'Re-audit baseline if AIter implementation becomes available'
        return moe_fused_gate(scores,bias,topk=6,scoring_func='sqrtsoftplus',
                              renormalize=True,routed_scaling_factor=1.5,
                              apply_routed_scaling_factor_on_output=False)
    for fn in (current,lambda:gfx90a_sqrtsoftplus_router(scores,bias,1.5,False)):
        assert fn() is not None
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):out=fn()
        graphs.append(g);outputs.append(out)
    ids_exact=weights_exact=0;max_abs=0.;stable=True
    for iteration in range(100):
        scores.normal_()
        if iteration%10==0:scores.zero_()  # equal logits exercise bias ties
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        ids_exact+=int(torch.equal(outputs[0][1],outputs[1][1]))
        weights_exact+=int(torch.equal(outputs[0][0],outputs[1][0]))
        max_abs=max(max_abs,float((outputs[0][0]-outputs[1][0]).abs().max()))
        assert all(torch.isfinite(o[0]).all() for o in outputs)
        saved=[v.clone() for v in outputs[1]]
        for _ in range(10):graphs[1].replay()
        torch.cuda.synchronize()
        stable &= all(torch.equal(v,s) for v,s in zip(outputs[1],saved))
    samples=[[],[]]
    for _ in range(7):
        for arm in (0,1,1,0):
            for _ in range(20):graphs[arm].replay()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(200):graphs[arm].replay()
            end.record();end.synchronize()
            samples[arm].append(start.elapsed_time(end)*5)
    print(json.dumps(dict(fixture='synthetic BF16 logits, real layer20 BF16 bias',
                          torch=str(torch.__version__),hip=str(torch.version.hip),
                          ids_exact=ids_exact,weights_exact=weights_exact,mutations=100,
                          max_abs=max_abs,replay_stable=stable,samples_us=samples,
                          median_us=[statistics.median(s) for s in samples])),flush=True)


if __name__=='__main__':main()
