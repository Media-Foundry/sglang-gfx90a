#!/usr/bin/env python3
"""Sweep TP4 gate R2 row-prefetch over learned-router layers of one real pass."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from scripts.rocm.bench_dsv4_tp4_gate_row_prefetch_oracle import (
    _jit_row_prefetch,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_gate_up_grouped_dpp,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    per_token_group_quant_int8,
)

E, M, T, I, K = 256, 32, 6, 512, 4096
A, R, W, G, LUT = 4, 2, 8, 2080, 2


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed(xs: list[float]) -> float:
    ys = sorted(xs)
    return statistics.mean(ys[1:-1]) if len(ys) > 2 else statistics.mean(ys)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1787803355.1855972.pt",
    )
    p.add_argument("--pass-index", type=int, default=37)
    p.add_argument("--recorded-world-size", type=int, default=8)
    p.add_argument("--layers", default="3-42")
    p.add_argument("--exact-layers", default="3,12,21,30,42")
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--csv", type=Path, default=Path("/tmp/dsv4_gate_row_prefetch_layers.csv"))
    p.add_argument("--summary", type=Path, default=Path("/tmp/dsv4_gate_row_prefetch_layers.json"))
    args = p.parse_args()

    lo, hi = (int(x) for x in args.layers.split("-"))
    layers = list(range(lo, hi + 1))
    exact_layers = {int(x) for x in args.exact_layers.split(",")}
    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)

    metadata = {}
    route_stats = {}
    for layer in layers:
        raw = payload["logical_count"][args.pass_index, layer]
        if torch.any(raw.remainder(args.recorded_world_size) != 0):
            raise RuntimeError(f"layer {layer}: counts not divisible by world size")
        counts = raw // args.recorded_world_size
        topk_ids = reconstruct_topk_from_counts(counts).cuda()
        md = make_metadata(topk_ids, assignments=A)
        metadata[layer] = md
        route_stats[layer] = {
            "active_experts": int((counts > 0).sum()),
            "assignments": int(counts.sum()),
            "a4_blocks": int(md.sorted_experts.numel()),
            "max_occupancy": int(counts.max()),
        }

    torch.manual_seed(20260830)
    x = torch.randn((M, K), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    w13 = torch.randint(0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, K // 32), 127, dtype=torch.uint8, device="cuda")
    out_a = torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda")
    out_b = torch.empty_like(out_a)
    baseline = _jit_gate_up_grouped_dpp(E, M, T, I, K, A, R, W, G, LUT)
    candidate = _jit_row_prefetch(0)

    def run(module, md, out):
        module.run(
            xq, xs, w13, s13, md.sorted_ids, md.sorted_experts, md.valid,
            out, 10.0,
        )

    # Five distributed layers, 100 bounded activation mutations each.
    mutation_x = torch.empty_like(x)
    for layer in sorted(exact_layers):
        md = metadata[layer]
        for mutation in range(args.mutations):
            mutation_x.normal_()
            mxq, mxs = per_token_group_quant_int8(mutation_x, 32)
            xq.copy_(mxq)
            xs.copy_(mxs)
            run(baseline, md, out_a)
            run(candidate, md, out_b)
            torch.cuda.synchronize()
            if not torch.equal(out_a, out_b):
                diff = (out_a.float() - out_b.float()).abs().max().item()
                raise RuntimeError(
                    f"layer={layer} mutation={mutation} mismatch max_abs={diff}"
                )
        print(f"EXACT layer={layer} mutations={args.mutations}", flush=True)

    # Restore one fixed input before route-only timing.
    mutation_x.normal_()
    mxq, mxs = per_token_group_quant_int8(mutation_x, 32)
    xq.copy_(mxq)
    xs.copy_(mxs)

    rows = []
    for layer in layers:
        md = metadata[layer]
        fn_a = lambda md=md: run(baseline, md, out_a)
        fn_b = lambda md=md: run(candidate, md, out_b)
        samples = {"A": [], "B": []}
        for _ in range(args.rounds):
            for name, fn in (("A", fn_a), ("B", fn_b), ("B", fn_b), ("A", fn_a)):
                samples[name].append(time_us(fn, args.warmup, args.iterations))
        a_us, b_us = trimmed(samples["A"]), trimmed(samples["B"])
        row = {
            "layer": layer,
            **route_stats[layer],
            "baseline_us": a_us,
            "candidate_us": b_us,
            "delta_us": b_us - a_us,
            "gain_pct": (a_us / b_us - 1.0) * 100.0,
        }
        rows.append(row)
        print("LAYER " + json.dumps(row, separators=(",", ":")), flush=True)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "recorder": str(Path(args.recorder).resolve()),
        "pass_index": args.pass_index,
        "layers": layers,
        "exact_layers": sorted(exact_layers),
        "exact_mutations_per_layer": args.mutations,
        "mean_delta_us": statistics.mean(r["delta_us"] for r in rows),
        "median_delta_us": statistics.median(r["delta_us"] for r in rows),
        "sum_delta_us": sum(r["delta_us"] for r in rows),
        "improved_layers": sum(r["delta_us"] < 0 for r in rows),
        "regressed_layers": sum(r["delta_us"] > 0 for r in rows),
        "worst_layers": sorted(rows, key=lambda r: r["delta_us"], reverse=True)[:8],
        "best_layers": sorted(rows, key=lambda r: r["delta_us"])[:8],
        "rows": rows,
    }
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY " + json.dumps({k: v for k, v in summary.items() if k != "rows"}, default=str), flush=True)


if __name__ == "__main__":
    main()
