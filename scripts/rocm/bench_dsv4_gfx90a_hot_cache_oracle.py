#!/usr/bin/env python3
"""Standalone TP8 BS32 learned-layer N64 w2 hot-cache mixed oracle.

The baseline is the current packed-FP4/LDS-LUT A4 down producer plus the fixed
reduction.  The candidate preserves the same metadata, FP32 partial layout and
reduction order, but selects a compact pre-expanded INT8 w2 cache per expert
block when the expert belongs to the layer's train-window top 64.  It has no
production selector.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_grouped
from sglang.kernels.ops.moe.gfx90a_fp4_hot_cache_oracle import (
    gfx90a_fp4_hot_cache_down_partial_oracle,
)


E = 256
M = 32
T = 6
N = 4096
K = 256
ASSIGNMENTS = 4
ROWS = 2
WAVES = 8
BLOCKS = 832
HOT_N = 64
LDS_LUT = 2


def unpack_fp4_i8(weight: torch.Tensor) -> torch.Tensor:
    lut = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.int8,
        device=weight.device,
    )
    out = torch.empty(
        (*weight.shape[:-1], weight.shape[-1] * 2),
        dtype=torch.int8,
        device=weight.device,
    )
    out[..., 0::2] = lut[(weight & 15).long()]
    out[..., 1::2] = lut[(weight >> 4).long()]
    return out


def reconstruct_topk_from_counts(counts: torch.Tensor) -> torch.Tensor:
    counts = counts.to(torch.int64).cpu()
    if counts.shape != (E,) or int(counts.sum()) != M * T:
        raise ValueError(f"expected [256] counts summing to 192, got {counts.shape}")
    rows: list[list[int]] = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True, stable=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [
                token
                for token in range(M)
                if len(rows[token]) < T and expert not in rows[token]
            ]
            if not choices:
                raise RuntimeError(f"cannot place expert {expert}")
            token = min(choices, key=lambda value: (len(rows[value]), value))
            rows[token].append(expert)
    result = torch.tensor(rows, dtype=torch.int32)
    if result.shape != (M, T) or any(len(set(row)) != T for row in rows):
        raise RuntimeError("invalid reconstructed top-k")
    return result


def make_metadata(topk_ids: torch.Tensor):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), ASSIGNMENTS):
            block = bucket[offset : offset + ASSIGNMENTS]
            ids.extend(block)
            ids.extend([M] * (ASSIGNMENTS - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def scan_hit_pct(counts: torch.Tensor, hot: torch.Tensor) -> float:
    blocks = torch.div(counts + 3, 4, rounding_mode="floor")
    return 100.0 * int(blocks[hot].sum()) / int(blocks.sum())


def select_layers_and_passes(analysis: dict, recorder_counts: torch.Tensor):
    rows = analysis["top_n"][str(HOT_N)]["per_layer"][3:]
    ordered = sorted(rows, key=lambda row: row["a4_scan_hit_pct"])
    p50 = analysis["top_n"][str(HOT_N)]["summaries"]["learned"][
        "a4_scan_hit_pct_per_layer"
    ]["p50"]
    chosen = {
        "low": ordered[0],
        "p50": min(rows, key=lambda row: abs(row["a4_scan_hit_pct"] - p50)),
        "high": ordered[-1],
    }
    test_indices = analysis["metadata"]["selected_raw_indices"][64:]
    result = []
    for label, row in chosen.items():
        layer = row["layer"]
        hot = torch.tensor(row["hot_expert_ids"], dtype=torch.int64)
        candidates = []
        for raw_index in test_indices:
            raw = recorder_counts[raw_index, layer]
            if torch.any(raw.remainder(8) != 0):
                raise RuntimeError("recorder count is not divisible by TP8")
            counts = raw // 8
            hit = scan_hit_pct(counts, hot)
            candidates.append((abs(hit - row["a4_scan_hit_pct"]), raw_index, hit))
        _, raw_index, pass_hit = min(candidates)
        result.append((label, row, raw_index, pass_hit))
    return result


def time_segment(fn, *, warmup: int, iterations: int) -> float:
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


def abba(fn_a, fn_b, *, warmup: int, iterations: int, rounds: int):
    a: list[float] = []
    b: list[float] = []
    for _ in range(rounds):
        a.append(time_segment(fn_a, warmup=warmup, iterations=iterations))
        b.append(time_segment(fn_b, warmup=warmup, iterations=iterations))
        b.append(time_segment(fn_b, warmup=warmup, iterations=iterations))
        a.append(time_segment(fn_a, warmup=warmup, iterations=iterations))
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--analysis", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")

    analysis = json.loads(Path(args.analysis).read_text())
    recorder = torch.load(args.recorder, map_location="cpu", weights_only=False)
    recorder_counts = recorder["logical_count"]
    selections = select_layers_and_passes(analysis, recorder_counts)

    torch.manual_seed(7)
    device = torch.device("cuda")
    w2 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device=device)
    baseline_module = _jit_down_grouped(
        E, M, T, N, K, ASSIGNMENTS, ROWS, WAVES, BLOCKS, LDS_LUT
    )
    results = []

    for label, row, raw_index, pass_hit in selections:
        layer = row["layer"]
        hot_ids_cpu = torch.tensor(row["hot_expert_ids"], dtype=torch.int64)
        hot_ids = hot_ids_cpu.to(device)
        expert_to_cache = torch.full((E,), -1, dtype=torch.int32, device=device)
        expert_to_cache[hot_ids] = torch.arange(HOT_N, dtype=torch.int32, device=device)

        torch.cuda.synchronize()
        allocated_before = torch.cuda.memory_allocated()
        reserved_before = torch.cuda.memory_reserved()
        build_begin = time.perf_counter_ns()
        hot_packed = w2.index_select(0, hot_ids)
        hot_weight = unpack_fp4_i8(hot_packed)
        del hot_packed
        torch.cuda.synchronize()
        build_us = (time.perf_counter_ns() - build_begin) / 1000.0
        allocated_delta = torch.cuda.memory_allocated() - allocated_before
        reserved_delta = torch.cuda.memory_reserved() - reserved_before

        counts = recorder_counts[raw_index, layer] // 8
        topk_ids = reconstruct_topk_from_counts(counts).to(device)
        sorted_ids, sorted_experts, valid = make_metadata(topk_ids)
        block_hot = expert_to_cache[sorted_experts] >= 0
        measured_block_hit = 100.0 * int(block_hot.sum()) / sorted_experts.numel()
        if abs(measured_block_hit - pass_hit) > 1e-9:
            raise RuntimeError("metadata and held-out scan hit disagree")

        xq = torch.randint(-127, 128, (M, T, K), dtype=torch.int8, device=device)
        x_scale = torch.rand((M, T, K // 32), dtype=torch.float32, device=device)
        topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
        partial_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
        partial_b = torch.empty_like(partial_a)
        out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
        out_b = torch.empty_like(out_a)

        def run_a() -> None:
            baseline_module.run_partial(
                xq,
                x_scale,
                w2,
                s2,
                sorted_ids,
                sorted_experts,
                valid,
                topk_weights,
                partial_a,
            )
            baseline_module.reduce(partial_a, out_a)

        def run_b() -> None:
            gfx90a_fp4_hot_cache_down_partial_oracle(
                xq,
                x_scale,
                w2,
                hot_weight,
                s2,
                expert_to_cache,
                sorted_ids,
                sorted_experts,
                valid,
                topk_weights,
                partial_b,
            )
            baseline_module.reduce(partial_b, out_b)

        run_a()
        run_b()
        torch.cuda.synchronize()
        if not torch.equal(partial_a, partial_b):
            raise AssertionError(
                f"layer {layer} FP32 partial mismatch "
                f"max_abs={(partial_a - partial_b).abs().max().item()}"
            )
        if not torch.equal(out_a, out_b):
            raise AssertionError(f"layer {layer} final BF16 mismatch")
        for replay in range(args.correctness_replays):
            xq.random_(-127, 128)
            x_scale.uniform_(1e-4, 0.1)
            topk_weights.uniform_(0.0, 1.0)
            run_a()
            run_b()
            torch.cuda.synchronize()
            if not torch.equal(partial_a, partial_b) or not torch.equal(out_a, out_b):
                raise AssertionError(f"layer {layer} replay {replay} mismatch")

        a_samples, b_samples = abba(
            run_a,
            run_b,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        a_median = statistics.median(a_samples)
        b_median = statistics.median(b_samples)
        result = {
            "label": label,
            "layer": layer,
            "raw_pass": raw_index,
            "heldout_layer_a4_hit_pct": row["a4_scan_hit_pct"],
            "selected_pass_a4_hit_pct": pass_hit,
            "active_experts": int((counts > 0).sum()),
            "a4_blocks": int(sorted_experts.numel()),
            "hot_a4_blocks": int(block_hot.sum()),
            "cache_build_us": build_us,
            "cache_tensor_bytes": hot_weight.numel() * hot_weight.element_size(),
            "allocated_delta_bytes": allocated_delta,
            "reserved_delta_bytes": reserved_delta,
            "correctness_replays": args.correctness_replays,
            "partial_exact": True,
            "final_exact": True,
            "baseline_us": a_median,
            "mixed_us": b_median,
            "saved_us": a_median - b_median,
            "speedup_pct": 100.0 * (a_median / b_median - 1.0),
            "baseline_samples_us": a_samples,
            "mixed_samples_us": b_samples,
            "passes_10us_gate": a_median - b_median >= 10.0,
        }
        results.append(result)
        print(json.dumps(result), flush=True)
        del hot_weight, expert_to_cache, partial_a, partial_b, out_a, out_b

    payload = {
        "format": "dsv4-tp8-bs32-n64-w2-hot-cache-oracle-v1",
        "recorder": str(Path(args.recorder).resolve()),
        "analysis": str(Path(args.analysis).resolve()),
        "geometry": {"A": 4, "rows": 2, "waves": 8, "blocks": 832},
        "results": results,
        "all_pass_10us_gate": all(row["passes_10us_gate"] for row in results),
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
