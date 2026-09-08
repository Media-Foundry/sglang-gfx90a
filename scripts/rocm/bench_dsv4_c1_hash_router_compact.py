#!/usr/bin/env python3
"""C1 hash-router compact-weight bound; no production change or persistent cache."""
import json
import statistics
import torch
from bench_dsv4_tp4_m32_paged_decode_geometry import capture, time_graph
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import gfx90a_wave64_bf16_gemv


def main():
    torch.manual_seed(20260908)
    x=torch.randn((1,4096),device='cuda',dtype=torch.bfloat16)
    w=torch.randn((256,4096),device='cuda',dtype=torch.bfloat16)
    ids=torch.tensor([3,27,51,89,177,241,0,1],device='cuda')
    compact=w.index_select(0,ids)
    funcs=[lambda:gfx90a_wave64_bf16_gemv(x,w),
           lambda:gfx90a_wave64_bf16_gemv(x,compact),
           lambda:gfx90a_wave64_bf16_gemv(x,w.index_select(0,ids))]
    graphs=[];outputs=[]
    for fn in funcs:
        assert fn() is not None
        torch.cuda.synchronize()
        g,out=capture(fn);graphs.append(g);outputs.append(out)
    exact=[0,0]
    for n in range(100):
        x.normal_();w.normal_()
        ids.copy_(torch.randperm(256,device='cuda')[:8])
        compact.copy_(w.index_select(0,ids))
        for g in graphs:g.replay()
        torch.cuda.synchronize()
        reference=outputs[0].index_select(1,ids)[:,:6]
        for i,out in enumerate(outputs[1:]):
            exact[i]+=int(torch.equal(reference,out[:,:6]))
    assert exact==[100,100]
    for g,out in zip(graphs,outputs):
        expected=out.clone()
        for _ in range(1000):g.replay()
        torch.cuda.synchronize();assert torch.equal(expected,out)
    samples=[[],[],[]]
    for _ in range(5):
        for i in (0,1,2,2,1,0):samples[i].append(time_graph(graphs[i],20,100))
    print(json.dumps(dict(arms=['full256','ideal_preselected8','gather8_then_gemv'],
        selected_exact=exact,mutations=100,stable_replays=1000,samples_us=samples,
        trimmed_us=[statistics.mean(sorted(s)[1:-1]) for s in samples])),flush=True)


if __name__=='__main__':main()
