#!/usr/bin/env python3
"""Isolated final HC-head geometry screen; no production dispatch changes.

Run on an idle GPU. Fixed K512 preserves the sequential K reduction blocks.
Synthetic mutations establish local parity, not full-model correctness.
"""
import json
import statistics

import torch
from sglang.kernels.ops.layernorm.mhc_head import _hc_head_kernel


def main():
    torch.manual_seed(20908)
    results = []
    for m in (1, 32):
        x = torch.randn(m, 4, 4096, device='cuda', dtype=torch.bfloat16)
        fn = torch.randn(4, 16384, device='cuda') * 0.01
        scale = torch.ones(1, device='cuda')
        base = torch.randn(4, device='cuda')
        a = torch.empty(m, 4096, device='cuda', dtype=torch.bfloat16)
        b = torch.empty_like(a)

        def run(out, warps, block_d):
            _hc_head_kernel[(m,)](x, fn, scale, base, out,
                hidden_size=4096, HC_MULT=4, K_TOTAL=16384,
                BLOCK_K=512, BLOCK_D=block_d, norm_eps=1e-6,
                hc_eps=1e-6, num_warps=warps)

        for warps, block_d in ((4, 512), (4, 1024), (8, 512), (8, 1024)):
            run(a, 4, 512)
            run(b, warps, block_d)
            torch.cuda.synchronize()
            ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
            with torch.cuda.graph(ga):
                run(a, 4, 512)
            with torch.cuda.graph(gb):
                run(b, warps, block_d)
            exact, stable, max_abs = 0, 0, 0.0
            for _ in range(100):
                x.normal_()
                fn.normal_(std=0.01)
                base.normal_()
                scale.uniform_(0.2, 2.0)
                ga.replay(); gb.replay()
                exact += int(torch.equal(a, b))
                max_abs = max(max_abs, (a.float() - b.float()).abs().max().item())
                witness = b.clone()
                for _ in range(10):
                    gb.replay()
                    stable += int(torch.equal(witness, b))
            timings = {'A': [], 'B': []}
            for _ in range(5):
                for name, graph in (('A', ga), ('B', gb), ('B', gb), ('A', ga)):
                    for _ in range(20): graph.replay()
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(200): graph.replay()
                    end.record(); end.synchronize()
                    timings[name].append(start.elapsed_time(end) * 1000 / 200)
            row = dict(m=m, warps=warps, block_d=block_d, exact=exact,
                       stable=stable, max_abs=max_abs, samples_us=timings,
                       median_us={k: statistics.median(v) for k, v in timings.items()})
            results.append(row)
            print(json.dumps(row), flush=True)
    print(json.dumps({'results': results, 'production_changed': False}), flush=True)


if __name__ == '__main__':
    main()
