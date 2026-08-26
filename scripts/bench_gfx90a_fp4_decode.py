#!/usr/bin/env python3
"""Microbenchmark the gfx90a grouped direct-FP4 decode kernels.

The default M=32 shapes match a TP8 DeepSeek-V4-Flash decode step.  This is a
kernel-development tool: it includes AIter sorting and group-32 activation
quantization in the full-stage timing, and verifies every candidate against the
first geometry before reporting latency.
"""

from __future__ import annotations

import argparse
import itertools
import statistics

import torch
from aiter.fused_moe import moe_sorting

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down_grouped,
    gfx90a_fp4_expert_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E = 256
TOPK = 6
HIDDEN = 4096
INTERMEDIATE = 256


def timed_us(fn, warmup: int, iterations: int, rounds: int):
    result = None
    samples = []
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
        samples.append(begin.elapsed_time(end) * 1000.0 / iterations)
    return statistics.median(samples), samples, result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=32)
    parser.add_argument("--assignments", type=int, default=4)
    parser.add_argument("--rows", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument(
        "--down-rows",
        type=int,
        nargs="+",
        help="down-stage rows; defaults to matching --rows",
    )
    parser.add_argument("--waves", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--blocks", type=int, nargs="+", default=[416, 624, 832])
    parser.add_argument(
        "--down-blocks",
        type=int,
        nargs="+",
        help="down-stage blocks; defaults to matching --blocks",
    )
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
    x = torch.randn((args.m, HIDDEN), dtype=torch.bfloat16, device=device)
    topk_ids = torch.randint(0, E, (args.m, TOPK), dtype=torch.int32, device=device)
    topk_weights = torch.rand((args.m, TOPK), dtype=torch.float32, device=device)
    w13 = torch.randint(
        0, 256, (E, 2 * INTERMEDIATE, HIDDEN // 2), dtype=torch.uint8, device=device
    )
    w2 = torch.randint(
        0, 256, (E, HIDDEN, INTERMEDIATE // 2), dtype=torch.uint8, device=device
    )
    s13 = torch.full(
        (E, 2 * INTERMEDIATE, HIDDEN // 32), 127, dtype=torch.uint8, device=device
    )
    s2 = torch.full(
        (E, HIDDEN, INTERMEDIATE // 32), 127, dtype=torch.uint8, device=device
    )
    xq, xs = per_token_group_quant_int8(x, 32)
    sorted_ids, _, sorted_experts, valid, _ = moe_sorting(
        topk_ids,
        topk_weights,
        E,
        HIDDEN,
        x.dtype,
        block_size=args.assignments,
    )

    reference = None
    row_pairs = (
        list(itertools.product(args.rows, args.down_rows))
        if args.down_rows is not None
        else [(rows, rows) for rows in args.rows]
    )
    block_pairs = (
        list(itertools.product(args.blocks, args.down_blocks))
        if args.down_blocks is not None
        else [(blocks, blocks) for blocks in args.blocks]
    )
    for (rows, down_rows), waves, (blocks, down_blocks) in itertools.product(
        row_pairs, args.waves, block_pairs
    ):
                def run():
                    intermediate = gfx90a_fp4_expert_gate_up_grouped(
                        xq,
                        xs,
                        w13,
                        s13,
                        sorted_ids,
                        sorted_experts,
                        valid,
                        TOPK,
                        10.0,
                        assignments=args.assignments,
                        rows=rows,
                        waves=waves,
                        blocks=blocks,
                    )
                    iq, isc = per_token_group_quant_int8(intermediate, 32)
                    return gfx90a_fp4_expert_down_grouped(
                        iq,
                        isc,
                        w2,
                        s2,
                        sorted_ids,
                        sorted_experts,
                        valid,
                        topk_weights,
                        assignments=args.assignments,
                        rows=down_rows,
                        waves=waves,
                        blocks=down_blocks,
                    )

                median, samples, output = timed_us(
                    run, args.warmup, args.iterations, args.rounds
                )
                if reference is None:
                    reference = output.clone()
                exact = torch.equal(reference, output)
                print(
                    f"m={args.m} assignments={args.assignments} "
                    f"gate_rows={rows} down_rows={down_rows} "
                    f"waves={waves} gate_blocks={blocks} "
                    f"down_blocks={down_blocks} median_us={median:.3f} "
                    f"samples={[round(x, 3) for x in samples]} exact={exact}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
