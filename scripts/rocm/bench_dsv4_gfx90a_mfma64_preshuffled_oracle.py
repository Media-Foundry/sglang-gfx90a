#!/usr/bin/env python3
"""ABBA raw versus AIter-preshuffled MFMA64 routed-MoE oracle."""

import argparse
import statistics
import time

import torch
from aiter.fused_moe import moe_sorting
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down_mfma32,
    gfx90a_fp4_expert_gate_up_mfma32,
)
from sglang.kernels.ops.quantization.gfx90a_int8_quant import (
    gfx90a_int8_group32_quant,
)


def timed_us(fn, iterations: int) -> float:
    for _ in range(3):
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
    parser.add_argument("--tokens", type=int, default=2304)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--gate-blocks", type=int, default=416)
    parser.add_argument("--down-blocks", type=int, default=312)
    parser.add_argument("--arm", choices=("raw", "preshuffled", "both"), default="both")
    args = parser.parse_args()
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")

    e, m, t, h, i, n = 256, args.tokens, 6, 4096, 1024, 4096
    generator = torch.Generator(device="cpu").manual_seed(20260902)
    raw_w13 = torch.randint(0, 256, (e, 2 * i, h // 2), dtype=torch.uint8,
                            generator=generator).cuda()
    raw_s13 = torch.randint(110, 114, (e * 2 * i, h // 32), dtype=torch.uint8,
                            generator=generator).cuda()
    raw_w2 = torch.randint(0, 256, (e, n, i // 2), dtype=torch.uint8,
                           generator=generator).cuda()
    raw_s2 = torch.randint(110, 114, (e * n, i // 32), dtype=torch.uint8,
                           generator=generator).cuda()
    shuffled_w13 = shuffle_weight_a16w4(raw_w13, 16, True)
    shuffled_s13 = shuffle_scale_a16w4(raw_s13, e, True)
    shuffled_w2 = shuffle_weight_a16w4(raw_w2, 16, False)
    shuffled_s2 = shuffle_scale_a16w4(raw_s2, e, False)
    # Match the router contract: experts within one token's top-k are unique.
    topk_ids = torch.rand((m, e), device="cuda").topk(t, dim=1).indices.to(
        torch.int32
    )
    topk_weights = torch.rand((m, t), dtype=torch.float32, device="cuda")
    x = torch.randn((m, h), dtype=torch.bfloat16, device="cuda")

    sorted_ids, _, sorted_experts, valid, _ = moe_sorting(
        topk_ids, topk_weights, e, h, torch.bfloat16, block_size=64
    )

    def run(preshuffled: bool):
        xq, xs = gfx90a_int8_group32_quant(x)
        mid = gfx90a_fp4_expert_gate_up_mfma32(
            xq, xs, shuffled_w13 if preshuffled else raw_w13, shuffled_s13,
            sorted_ids, sorted_experts, valid, t, 10.0, blocks=args.gate_blocks,
            broadcast_scales=1, assignments=64, preshuffled=preshuffled,
        )
        iq, isc = gfx90a_int8_group32_quant(mid)
        out = gfx90a_fp4_expert_down_mfma32(
            iq, isc, shuffled_w2 if preshuffled else raw_w2, shuffled_s2,
            sorted_ids, sorted_experts, valid, topk_weights,
            blocks=args.down_blocks,
            broadcast_scales=1, assignments=64, preshuffled=preshuffled,
        )
        return mid, iq, isc, out

    if args.arm == "both":
        for mutation in range(args.mutations):
            x.normal_()
            topk_weights.uniform_()
            a = run(False)
            torch.cuda.synchronize()
            b = run(True)
            torch.cuda.synchronize()
            for name, lhs, rhs in zip(("mid", "iq", "scale", "out"), a, b):
                if not torch.equal(lhs, rhs):
                    raise RuntimeError(
                        f"mutation={mutation} tensor={name} max_abs="
                        f"{(lhs.float() - rhs.float()).abs().max().item()}"
                    )
        print(f"CORRECTNESS mutations={args.mutations} all_exact=True")

    arms = ("raw", "preshuffled") if args.arm == "both" else (args.arm,)
    values = {arm: [] for arm in arms}
    for _ in range(args.rounds):
        order = arms + tuple(reversed(arms))
        for arm in order:
            values[arm].append(timed_us(lambda a=arm: run(a == "preshuffled"),
                                        args.iterations))
    for arm, samples in values.items():
        trimmed = sorted(samples)[1:-1] if len(samples) > 2 else samples
        print(f"RESULT arm={arm} median_us={statistics.median(samples):.3f} "
              f"trimmed_us={statistics.mean(trimmed):.3f} samples={samples}")


if __name__ == "__main__":
    main()
