#!/usr/bin/env python3
"""Standalone AIter CKTile FP4 grouped-MoE benchmark for DSV4 TP4 shapes."""

from __future__ import annotations

import argparse
import importlib
import statistics

import torch
from aiter.fused_moe import ActivationType, QuantType, fused_moe
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4


def timed_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=4608)
    parser.add_argument("--block-m", type=int, choices=(16, 32, 64, 128), default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")
    fused_moe_module = importlib.import_module("aiter.fused_moe")
    fused_moe_module.fused_moe_1stage_dict.setdefault("gfx90a", set())

    e, m, t, h, i = 256, args.tokens, 6, 4096, 512
    g = torch.Generator(device="cpu").manual_seed(20260902 + m)
    raw_w13 = torch.randint(
        0, 256, (e, 2 * i, h // 2), dtype=torch.uint8, generator=g
    ).cuda()
    raw_s13 = torch.randint(
        110, 114, (e * 2 * i, h // 32), dtype=torch.uint8, generator=g
    ).cuda()
    raw_w2 = torch.randint(
        0, 256, (e, h, i // 2), dtype=torch.uint8, generator=g
    ).cuda()
    raw_s2 = torch.randint(
        110, 114, (e * h, i // 32), dtype=torch.uint8, generator=g
    ).cuda()
    w13 = shuffle_weight_a16w4(raw_w13, 16, True)
    s13 = shuffle_scale_a16w4(raw_s13, e, True)
    w2 = shuffle_weight_a16w4(raw_w2, 16, False)
    s2 = shuffle_scale_a16w4(raw_s2, e, False)
    fp4 = getattr(torch, "float4_e2m1fn_x2", None)
    if fp4 is not None:
        w13 = w13.view(fp4)
        w2 = w2.view(fp4)
    w13.is_shuffled = True
    w2.is_shuffled = True

    x = torch.randn((m, h), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.rand((m, e), device="cuda").topk(t, dim=1).indices.to(
        torch.int32
    )
    topk_weights = torch.rand((m, t), dtype=torch.float32, device="cuda")
    out = torch.empty((m, h), dtype=torch.bfloat16, device="cuda")

    def run():
        return fused_moe(
            x,
            w13,
            w2,
            topk_weights,
            topk_ids,
            activation=ActivationType.Dsv4Silu,
            quant_type=QuantType.per_1x32,
            w1_scale=s13,
            w2_scale=s2,
            block_size_M=args.block_m,
            dtype=torch.bfloat16,
            hidden_pad=0,
            intermediate_pad=0,
            splitk=0,
            moe_out=out,
            preshuffle=True,
        )

    witness = run().clone()
    replay = run()
    torch.cuda.synchronize()
    if not torch.isfinite(replay).all():
        raise RuntimeError("non-finite output")
    print(f"REPLAY exact={torch.equal(witness, replay)}")
    samples = [timed_us(run, args.warmup, args.iterations) for _ in range(args.rounds)]
    trimmed = sorted(samples)[1:-1] if len(samples) > 2 else samples
    print(
        f"RESULT tokens={m} block_m={args.block_m} "
        f"median_us={statistics.median(samples):.3f} "
        f"trimmed_us={statistics.mean(trimmed):.3f} samples={samples}"
    )


if __name__ == "__main__":
    main()
