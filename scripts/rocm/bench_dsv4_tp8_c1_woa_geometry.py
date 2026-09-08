#!/usr/bin/env python3
"""C1 G1 wo_a geometry oracle; fixed unroll preserves the reduction tree."""
import argparse
import json
import statistics
from pathlib import Path

import torch

from sglang.kernels.jit.utils import load_jit, make_cpp_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--direct-x", action="store_true",
                   help="requires the experimental kStageX HIP template parameter")
    a = p.parse_args()
    torch.manual_seed(20908)
    configs = [(1, 4), (1, 2), (1, 8), (2, 2), (2, 4), (2, 8)]
    if a.direct_x:
        configs = [(1, 4), (1, 4), (1, 8), (2, 4)]
    # Rotating 43 independent layer weights avoids timing only L2-resident B.
    weights = [torch.randn(1, 1024, 4096, device="cuda", dtype=torch.bfloat16)
               for _ in range(43)]
    x = torch.randn(1, 1, 4096, device="cuda", dtype=torch.bfloat16)
    modules, outputs, graphs = {}, {}, {}
    for index, (rows, waves) in enumerate(configs):
        direct = a.direct_x and index > 0
        name = f"r{rows}u2w{waves}" + ("_direct" if direct else "")
        template = (1, 1, 1024, 4096, rows, 2, waves)
        args = make_cpp_args(*template, False) if direct else make_cpp_args(*template)
        modules[name] = load_jit(
            "gfx90a_bf16_grouped_gemv", *args,
            cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
            cuda_wrappers=[("run", f"sglang::Gfx90aBf16GroupedGemvKernel<{args}>::run")],
            extra_cuda_cflags=["-O3"],
        )
        outputs[name] = torch.empty(1, 1, 1024, device="cuda", dtype=torch.bfloat16)
        mod, out = modules[name], outputs[name]
        for _ in range(3):
            mod.run(x, weights[0], out)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            mod.run(x, weights[0], out)
        graphs[name] = g
        print("built", name, flush=True)
    correctness = {n: {"exact": 0, "replay_exact": 0, "finite": 0} for n in modules}
    for i in range(100):
        x.normal_()
        if i % 25 == 0:
            weights[0].normal_()
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        ref = outputs["r1u2w4"].clone()
        for n, g in graphs.items():
            before = outputs[n].clone()
            for _ in range(10):
                g.replay()
            correctness[n]["exact"] += int(torch.equal(ref, outputs[n]))
            correctness[n]["replay_exact"] += int(torch.equal(before, outputs[n]))
            correctness[n]["finite"] += int(torch.isfinite(outputs[n]).all())
    results = {"correctness": correctness, "samples_us": {}, "median_us": {}}
    for mode in ("same_weight", "43_weights"):
        bursts = {}
        for n, mod in modules.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for j in range(43):
                    mod.run(x, weights[j if mode == "43_weights" else 0], outputs[n])
            bursts[n] = g
        samples = {n: [] for n in modules}
        baseline = "r1u2w4"
        for candidate in modules:
            if candidate == baseline:
                continue
            for n in [baseline, candidate, candidate, baseline] * 3:
                bursts[n].replay()
                start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(10):
                    bursts[n].replay()
                end.record()
                end.synchronize()
                samples[n].append(start.elapsed_time(end) * 1000 / (43 * 10))
        results["samples_us"][mode] = samples
        results["median_us"][mode] = {n: statistics.median(v) for n, v in samples.items()}
        a.output.write_text(json.dumps(results, indent=2) + "\n")
        print(mode, results["median_us"][mode], flush=True)
    assert all(v["exact"] == v["replay_exact"] == v["finite"] == 100
               for v in correctness.values()), correctness


if __name__ == "__main__":
    main()
