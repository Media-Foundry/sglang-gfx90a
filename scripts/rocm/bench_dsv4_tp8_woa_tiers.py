#!/usr/bin/env python3
"""Isolated G1 wo_a tier screen; does not change the production selector."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    _jit_gfx90a_bf16_grouped_gemv_module,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.manual_seed(20908)
    results = {}
    for m in (1, 2, 4, 8):
        x = torch.randn(m, 1, 4096, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
        out = torch.empty(m, 1, 1024, device="cuda", dtype=torch.bfloat16)
        module = _jit_gfx90a_bf16_grouped_gemv_module(m, 1)

        def candidate():
            module.run(x, w, out)
            return out

        funcs = {"einsum": lambda: torch.einsum("tgd,grd->tgr", x, w),
                 "grouped": candidate}
        graphs = {}
        for name, fn in funcs.items():
            for _ in range(3):
                fn()
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                y = fn()
            graphs[name] = (graph, y)
        mutations = []
        for i in range(100):
            x.normal_()
            if i % 25 == 0:
                w.normal_()
            for graph, _ in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
            before = out.clone()
            for _ in range(10):
                graphs["grouped"][0].replay()
            ref = torch.matmul(x.float(), w[0].float().T)
            mutations.append({
                "replay_exact": torch.equal(before, out),
                "baseline_exact": torch.equal(graphs["einsum"][1], out),
                "finite": bool(torch.isfinite(out).all()),
                "relative_l2": float((out.float() - ref).norm() / ref.norm()),
            })
        bursts = {}
        for name, fn in funcs.items():
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(64):
                    fn()
            bursts[name] = graph
        samples = {name: [] for name in funcs}
        for name in ["einsum", "grouped", "grouped", "einsum"] * 5:
            bursts[name].replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            for _ in range(20):
                bursts[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / (64 * 20))
        results[str(m)] = {
            "samples_us": samples,
            "median_us": {k: statistics.median(v) for k, v in samples.items()},
            "mutations": mutations,
        }
        args.output.write_text(json.dumps(results, indent=2) + "\n")
        print(m, results[str(m)]["median_us"], flush=True)


if __name__ == "__main__":
    main()
