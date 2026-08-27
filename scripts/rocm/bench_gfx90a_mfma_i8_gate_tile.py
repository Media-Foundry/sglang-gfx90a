#!/usr/bin/env python3
"""ABBA the real-shape A4xN64xK4096 MFMA tile against current sdot."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import (
    _jit_gate_tile_module,
)


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


def abba(fn_a, fn_b, warmup: int, iterations: int, rounds: int):
    a, b = [], []
    for _ in range(rounds):
        a.append(time_us(fn_a, warmup, iterations))
        b.append(time_us(fn_b, warmup, iterations))
        b.append(time_us(fn_b, warmup, iterations))
        a.append(time_us(fn_a, warmup, iterations))
    return a, b


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=20)
    parser.add_argument("--output")
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")

    torch.manual_seed(23)
    device = torch.device("cuda")
    xq = torch.randint(-127, 128, (4, 4096), dtype=torch.int8, device=device)
    xs = torch.rand((4, 128), dtype=torch.float32, device=device) * 0.05
    weight = torch.randint(0, 256, (64, 2048), dtype=torch.uint8, device=device)
    ws = torch.randint(124, 130, (64, 128), dtype=torch.uint8, device=device)
    module = _jit_gate_tile_module()
    ref_out = torch.empty((4, 64), dtype=torch.float32, device=device)
    ref_group = torch.empty((128, 4, 64), dtype=torch.int32, device=device)
    candidate_out = {
        split: torch.empty_like(ref_out) for split in (1, 2, 4, 8)
    }
    candidate_group = {
        split: torch.empty_like(ref_group) for split in (1, 2, 4, 8)
    }

    def run_ref_check():
        module.reference_check(xq, xs, weight, ws, ref_out, ref_group)

    def run_ref_timed():
        module.reference_timed(xq, xs, weight, ws, ref_out, ref_group)

    def run_candidate(split: int, mode: str):
        getattr(module, f"mfma_split{split}_{mode}")(
            xq,
            xs,
            weight,
            ws,
            candidate_out[split],
            candidate_group[split],
        )

    error_max = {split: 0.0 for split in (1, 2, 4, 8)}
    error_rel = {split: 0.0 for split in (1, 2, 4, 8)}
    for replay in range(args.correctness_replays):
        xq.random_(-127, 128)
        xs.uniform_(1e-4, 0.05)
        weight.random_(0, 256)
        ws.random_(124, 130)
        run_ref_check()
        for split in (1, 2, 4, 8):
            run_candidate(split, "check")
        torch.cuda.synchronize()
        for split in (1, 2, 4, 8):
            if not torch.equal(ref_group, candidate_group[split]):
                diff = (ref_group - candidate_group[split]).abs().max().item()
                raise AssertionError(
                    f"split={split} replay={replay} group int mismatch {diff}"
                )
            delta = (ref_out - candidate_out[split]).float()
            error_max[split] = max(error_max[split], delta.abs().max().item())
            rel = delta.norm().item() / max(ref_out.float().norm().item(), 1e-30)
            error_rel[split] = max(error_rel[split], rel)

    results = []
    for split in (1, 2, 4, 8):
        def candidate_timed(split=split):
            run_candidate(split, "timed")

        a, b = abba(
            run_ref_timed,
            candidate_timed,
            args.warmup,
            args.iterations,
            args.rounds,
        )
        a_med = statistics.median(a)
        b_med = statistics.median(b)
        row = {
            "split": split,
            "integer_group_exact": True,
            "correctness_replays": args.correctness_replays,
            "max_abs": error_max[split],
            "max_rel_l2": error_rel[split],
            "sdot_us": a_med,
            "mfma_us": b_med,
            "speedup": a_med / b_med,
            "passes_2x_gate": a_med / b_med >= 2.0,
            "sdot_samples_us": a,
            "mfma_samples_us": b,
        }
        results.append(row)
        print(json.dumps(row), flush=True)
    payload = {
        "format": "gfx90a-mfma-i8-a4-n64-k4096-tile-v1",
        "timed_writes_group_oracle": False,
        "results": results,
        "any_passes_2x_gate": any(row["passes_2x_gate"] for row in results),
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    if args.output:
        with open(args.output, "w") as handle:
            handle.write(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
