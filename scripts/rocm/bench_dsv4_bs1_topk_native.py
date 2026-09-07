#!/usr/bin/env python3
"""Small fixed-shape screen of production Triton vs existing wave64 TopK.

No server/profile changes. Runs on the single HIP_VISIBLE_DEVICES GPU. This
synthetic screen is not an E2E correctness or performance acceptance result.
"""
import argparse
import json
import os
from pathlib import Path
import statistics

os.environ["SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER"] = "0"
os.environ["SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER"] = "0"

import torch
from sglang.kernels.ops.moe.moe_fused_gate import moe_fused_gate
from sglang.kernels.ops.moe.gfx90a_grouped_router import gfx90a_sqrtsoftplus_router


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.manual_seed(20907)
    scores = torch.randn((1, 256), device="cuda", dtype=torch.bfloat16)
    bias = torch.randn(256, device="cuda", dtype=torch.bfloat16)
    funcs = {
        "triton": lambda: moe_fused_gate(
            scores, bias, topk=6, scoring_func="sqrtsoftplus", renormalize=True,
            routed_scaling_factor=1.5, apply_routed_scaling_factor_on_output=True,
        ),
        "hip": lambda: gfx90a_sqrtsoftplus_router(scores, bias, 1.5, True),
    }
    graphs = {}
    for name, fn in funcs.items():
        for _ in range(3):
            assert fn() is not None
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = fn()
        graphs[name] = (g, out)
    result = {"mutations": [], "timings_us": {name: [] for name in funcs}}
    for i in range(100):
        scores.normal_(0, 4)
        bias.normal_(0, 0.2)
        if i < 4:
            scores.fill_([0, -40, 40, 100][i])
            bias.zero_()
        for g, _ in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        aw, ai = graphs["triton"][1]
        bw, bi = graphs["hip"][1]
        stable = (bw.clone(), bi.clone())
        for _ in range(10):
            graphs["hip"][0].replay()
        torch.cuda.synchronize()
        result["mutations"].append({
            "i": i, "ids_exact": torch.equal(ai, bi),
            "weights_exact": torch.equal(aw, bw),
            "max_abs_weight_error": float((aw-bw).abs().max()),
            "replay_exact": torch.equal(stable[0], bw) and torch.equal(stable[1], bi),
            "finite": bool(torch.isfinite(bw).all()),
        })
    bursts = {}
    for name, fn in funcs.items():
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(128):
                fn()
        bursts[name] = g
    for name in ["triton", "hip", "hip", "triton"] * 3:
        bursts[name].replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            bursts[name].replay()
        end.record()
        end.synchronize()
        result["timings_us"][name].append(start.elapsed_time(end)*1000/(20*128))
    result["median_us"] = {k: statistics.median(v) for k,v in result["timings_us"].items()}
    result["summary"] = {
        k: sum(row[k] for row in result["mutations"])
        for k in ("ids_exact", "weights_exact", "replay_exact", "finite")
    }
    result["summary"]["max_abs_weight_error"] = max(r["max_abs_weight_error"] for r in result["mutations"])
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"median_us":result["median_us"],"correctness":result["summary"]}),flush=True)


if __name__ == "__main__":
    main()
