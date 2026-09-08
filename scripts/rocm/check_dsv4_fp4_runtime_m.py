#!/usr/bin/env python3
"""Exact runtime-M versus fixed-M grouped MoE, with ragged expert blocks."""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench_gfx90a_fp4_decode import make_sorted_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_gate_up_grouped as gate,
    gfx90a_fp4_expert_down_grouped as down,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", nargs="+", type=int, default=[129, 369, 511, 1023])
    a = p.parse_args()
    torch.manual_seed(20909)
    w13 = torch.randint(0, 256, (256, 512, 2048), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (256, 4096, 128), dtype=torch.uint8, device="cuda")
    s13 = torch.randint(119, 124, (256, 512, 128), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(119, 124, (256, 4096, 8), dtype=torch.uint8, device="cuda")
    records = []
    for m in a.rows:
        x = torch.randn(m, 4096, dtype=torch.bfloat16, device="cuda")
        ids = torch.randint(0, 256, (m, 6), dtype=torch.int32, device="cuda")
        sorted_ids, experts, valid = make_sorted_metadata(ids, 4)
        weights = torch.rand(m, 6, dtype=torch.float32, device="cuda")

        def run(dynamic):
            xq, xs = per_token_group_quant_int8(x, 32)
            mid = gate(xq, xs, w13, s13, sorted_ids, experts, valid, 6, 10.0,
                       assignments=4, rows=2, waves=8, blocks=416, runtime_m=dynamic)
            iq, isc = per_token_group_quant_int8(mid, 32)
            out = down(iq, isc, w2, s2, sorted_ids, experts, valid, weights,
                       assignments=4, rows=2, waves=8, blocks=312, runtime_m=dynamic)
            return mid, out

        graphs = {}
        for dynamic in (False, True):
            run(dynamic)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                result = run(dynamic)
            graphs[dynamic] = (g, result)
        checks = []
        for rep in range(10):
            x.normal_()
            weights.uniform_()
            if rep % 3 == 0:
                s13.random_(119, 124)
                s2.random_(119, 124)
            for g, _ in graphs.values():
                g.replay()
            torch.cuda.synchronize()
            exact = [torch.equal(u, v) for u, v in zip(graphs[False][1], graphs[True][1])]
            before = [v.clone() for v in graphs[True][1]]
            for _ in range(10):
                graphs[True][0].replay()
            stable = all(torch.equal(u, v) for u, v in zip(before, graphs[True][1]))
            finite = all(bool(torch.isfinite(v).all()) for v in graphs[True][1])
            checks.append({"exact": exact, "stable": stable, "finite": finite})
            assert all(exact) and stable and finite, (m, rep, checks[-1])
        samples = {False: [], True: []}
        for dynamic in [False, True, True, False] * 3:
            g = graphs[dynamic][0]
            g.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(10):
                g.replay()
            end.record()
            end.synchronize()
            samples[dynamic].append(start.elapsed_time(end) * 100)
        row = {"M": m, "checks": checks, "samples_us": samples,
               "median_us": {k: statistics.median(v) for k, v in samples.items()}}
        records.append(row)
        a.output.write_text(json.dumps(records, indent=2) + "\n")
        print(m, row["median_us"], flush=True)


if __name__ == "__main__":
    main()
