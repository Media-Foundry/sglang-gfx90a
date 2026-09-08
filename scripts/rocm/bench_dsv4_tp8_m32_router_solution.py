#!/usr/bin/env python3
"""Native-M32 router GEMM oracle, not a production selector."""
import json
import statistics
import torch
from aiter.tuned_gemm import hipb_gemm, tgemm
from bench_dsv4_tp4_m32_paged_decode_geometry import capture, time_graph


def main():
    torch.manual_seed(20260908)
    x = torch.randn((32,4096), device='cuda', dtype=torch.bfloat16)
    w = torch.randn((256,4096), device='cuda', dtype=torch.bfloat16)
    # Explicit warmup before graph capture initializes library workspaces.
    baseline = lambda: tgemm.mm(x, w, otype=x.dtype)
    candidate = lambda: hipb_gemm(x, w, 4358)
    for fn in (baseline, candidate):
        fn(); torch.cuda.synchronize()
    ga, a = capture(baseline)
    gb, b = capture(candidate)
    exact = 0
    max_abs = 0.
    max_rel = 0.
    for n in range(100):
        x.normal_(std=.5)
        if n % 10 == 0: w.normal_(std=.25)
        ga.replay(); gb.replay(); torch.cuda.synchronize()
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        exact += int(torch.equal(a,b))
        diff = a.float()-b.float()
        max_abs=max(max_abs,diff.abs().max().item())
        max_rel=max(max_rel,(diff.norm()/a.float().norm()).item())
    expected=b.clone()
    for _ in range(1000): gb.replay()
    torch.cuda.synchronize()
    assert torch.equal(expected,b)
    samples=[[],[]]
    for _ in range(5):
        for arm in (0,1,1,0):
            samples[arm].append(time_graph((ga,gb)[arm],20,100))
    print(json.dumps(dict(shape=[32,256,4096], solution=4358,
        exact_mutations=exact,mutations=100,max_abs=max_abs,max_relative_l2=max_rel,
        stable_replays=1000,samples_us=samples,
        trimmed_us=[statistics.mean(sorted(s)[1:-1]) for s in samples])),flush=True)


if __name__=='__main__': main()
