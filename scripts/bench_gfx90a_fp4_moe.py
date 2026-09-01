#!/usr/bin/env python3
"""Microbenchmark the gfx90a DSV4 FP4 routed-MoE prefill kernels.

The tensors and sorted-token metadata match TP4/EP1 DeepSeek-V4-Flash:
256 local experts, top-k 6, hidden 4096, and intermediate shard 512.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down_mfma32,
    gfx90a_fp4_expert_gate_up_mfma32,
)


E = 256
TOPK = 6
HIDDEN = 4096
INTERMEDIATE = 512


def make_sorted_metadata(m: int, block_size: int, device: torch.device):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token in range(m):
        for slot in range(TOPK):
            expert = (token * TOPK + slot) % E
            buckets[expert].append((slot << 24) | token)

    sorted_ids: list[int] = []
    sorted_experts: list[int] = []
    sentinel = m
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), block_size):
            block = bucket[offset : offset + block_size]
            sorted_ids.extend(block)
            sorted_ids.extend([sentinel] * (block_size - len(block)))
            sorted_experts.append(expert)

    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(sorted_experts, dtype=torch.int32, device=device),
        torch.tensor([len(sorted_ids), 0], dtype=torch.int32, device=device),
    )


def timed_ms(fn, warmup: int, iterations: int, rounds: int):
    samples = []
    result = None
    for _ in range(rounds):
        for _ in range(warmup):
            result = fn()
        torch.cuda.synchronize()
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(iterations):
            result = fn()
        end.record()
        end.synchronize()
        samples.append(begin.elapsed_time(end) / iterations)
    return samples, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("gate", "down"), required=True)
    parser.add_argument(
        "--m", type=int, choices=(512, 1024, 2048, 4096), default=2048
    )
    parser.add_argument("--blocks", type=int, required=True)
    parser.add_argument("--split", type=int, choices=(2, 4, 8), required=True)
    parser.add_argument("--broadcast-scales", type=int, choices=(0, 1), default=1)
    parser.add_argument("--assignments", type=int, choices=(32, 64), default=32)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("this benchmark requires ROCm")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"this benchmark requires gfx90a, got {arch}")

    torch.manual_seed(7)
    device = torch.device("cuda")
    sorted_ids, sorted_experts, num_valid = make_sorted_metadata(
        args.m, args.assignments, device
    )
    weight_scale_code = 127  # E8M0 scale 1.0

    if args.stage == "gate":
        xq = torch.randint(-8, 9, (args.m, HIDDEN), dtype=torch.int8, device=device)
        x_scale = torch.rand(
            (args.m, HIDDEN // 32), dtype=torch.float32, device=device
        )
        weight = torch.randint(
            0,
            256,
            (E, 2 * INTERMEDIATE, HIDDEN // 2),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = torch.full(
            (E, 2 * INTERMEDIATE, HIDDEN // 32),
            weight_scale_code,
            dtype=torch.uint8,
            device=device,
        )

        def run():
            return gfx90a_fp4_expert_gate_up_mfma32(
                xq,
                x_scale,
                weight,
                weight_scale,
                sorted_ids,
                sorted_experts,
                num_valid,
                TOPK,
                10.0,
                blocks=args.blocks,
                split=args.split,
                broadcast_scales=args.broadcast_scales,
                assignments=args.assignments,
            )

    else:
        xq = torch.randint(
            -8,
            9,
            (args.m, TOPK, INTERMEDIATE),
            dtype=torch.int8,
            device=device,
        )
        x_scale = torch.rand(
            (args.m, TOPK, INTERMEDIATE // 32),
            dtype=torch.float32,
            device=device,
        )
        weight = torch.randint(
            0,
            256,
            (E, HIDDEN, INTERMEDIATE // 2),
            dtype=torch.uint8,
            device=device,
        )
        weight_scale = torch.full(
            (E, HIDDEN, INTERMEDIATE // 32),
            weight_scale_code,
            dtype=torch.uint8,
            device=device,
        )
        topk_weights = torch.rand(
            (args.m, TOPK), dtype=torch.float32, device=device
        )

        def run():
            return gfx90a_fp4_expert_down_mfma32(
                xq,
                x_scale,
                weight,
                weight_scale,
                sorted_ids,
                sorted_experts,
                num_valid,
                topk_weights,
                blocks=args.blocks,
                split=args.split,
                broadcast_scales=args.broadcast_scales,
                assignments=args.assignments,
            )

    samples, result = timed_ms(run, args.warmup, args.iterations, args.rounds)
    print(
        f"stage={args.stage} m={args.m} blocks={args.blocks} split={args.split} "
        f"broadcast={args.broadcast_scales} assignments={args.assignments} samples_ms="
        + ",".join(f"{x:.6f}" for x in samples)
        + f" median_ms={statistics.median(samples):.6f} "
        f"finite={bool(torch.isfinite(result).all())}",
        flush=True,
    )


if __name__ == "__main__":
    main()
