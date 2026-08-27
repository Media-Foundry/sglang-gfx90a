#!/usr/bin/env python3
"""Standalone-only benchmark for the gfx90a M32 INT8 projection prototype."""

from __future__ import annotations

import argparse
import statistics
import time

import torch
import torch.nn.functional as F

from sglang.kernels.ops.quantization.gfx90a_int8_weight_gemm_m32 import (
    gfx90a_wave64_int8_weight_gemm_m32,
)


def quantize_weight(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = weight.float().abs().amax(dim=1).clamp_min(1.0e-12) / 127.0
    qweight = (weight.float() / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return qweight.contiguous(), scale.float().contiguous()


def timed_ms(fn, iterations: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iterations


def trimmed(values: list[float]) -> float:
    ordered = sorted(values)
    core = ordered[1:-1] if len(ordered) >= 5 else ordered
    return statistics.mean(core)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-tile", type=int, choices=(4, 8), default=8)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--input", type=str)
    parser.add_argument("--weight", type=str)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    if args.input and args.weight:
        x = torch.load(args.input, map_location="cuda", weights_only=True).to(torch.bfloat16)
        weight = torch.load(args.weight, map_location="cuda", weights_only=True).to(
            torch.bfloat16
        )
    else:
        torch.manual_seed(17)
        x = torch.randn((32, 4096), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((1536, 4096), device="cuda", dtype=torch.bfloat16) / 32
    x = x.contiguous()
    weight = weight.contiguous()
    assert x.shape == (32, 4096) and weight.shape == (1536, 4096)

    qweight, weight_scale = quantize_weight(weight)
    dequant_weight = (qweight.float() * weight_scale[:, None]).to(torch.bfloat16)
    reference = F.linear(x, dequant_weight)
    candidate = gfx90a_wave64_int8_weight_gemm_m32(
        x, qweight, weight_scale, a_tile=args.a_tile
    )
    assert candidate is not None
    replay = gfx90a_wave64_int8_weight_gemm_m32(
        x, qweight, weight_scale, a_tile=args.a_tile
    )
    assert replay is not None
    diff = candidate.float() - reference.float()
    print(
        f"shape={tuple(x.shape)}x{weight.shape[0]} a_tile={args.a_tile} "
        f"max_abs={diff.abs().max().item():.6g} "
        f"rel_l2={diff.norm().item() / reference.float().norm().item():.6g} "
        f"bitwise_replay={torch.equal(candidate, replay)}"
    )

    # Warm both paths, then alternate ABBA/BBAA by round to reduce drift.
    for _ in range(20):
        F.linear(x, dequant_weight)
        gfx90a_wave64_int8_weight_gemm_m32(x, qweight, weight_scale, a_tile=args.a_tile)
    torch.cuda.synchronize()
    ref_ms: list[float] = []
    cand_ms: list[float] = []
    for round_id in range(args.rounds):
        ref_fn = lambda: F.linear(x, dequant_weight)
        cand_fn = lambda: gfx90a_wave64_int8_weight_gemm_m32(
            x, qweight, weight_scale, a_tile=args.a_tile
        )
        order = (ref_fn, cand_fn, cand_fn, ref_fn)
        if round_id & 1:
            order = tuple(reversed(order))
        samples = [timed_ms(fn, args.iterations) for fn in order]
        ref_ms.append((samples[0] + samples[3]) / 2)
        cand_ms.append((samples[1] + samples[2]) / 2)
    ref = trimmed(ref_ms)
    cand = trimmed(cand_ms)
    print(
        f"reference_ms={ref:.6f} candidate_ms={cand:.6f} "
        f"speedup={ref / cand:.4f}x delta={(cand / ref - 1) * 100:+.2f}%"
    )


if __name__ == "__main__":
    main()
