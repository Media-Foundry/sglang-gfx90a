#!/usr/bin/env python3
"""Standalone MFMA64 I32-owner gate/quant oracle at prefill M2048/M2304.

Arm A is the production raw-weight MFMA64 gate followed by the exact standalone
group-32 quantizer.  Arm B computes the same two I16 tiles sequentially inside
one CTA and quantizes their local I32 group before publication.  The timing arm
does not materialize BF16 intermediates.  Nothing in this file changes a model
selector.

GPU convention for this repository: use physical GCD 4 for this single-GCD
oracle, for example ``HIP_VISIBLE_DEVICES=4 python ...``.
"""

from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from aiter.fused_moe import moe_sorting

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down_mfma32,
    gfx90a_fp4_expert_gate_up_mfma32,
)
from sglang.kernels.ops.moe.gfx90a_fp4_mfma64_gate_quant_oracle import (
    gfx90a_fp4_mfma64_gate_quant_oracle,
)
from sglang.kernels.ops.quantization.gfx90a_int8_quant import (
    gfx90a_int8_group32_quant,
)


E, T, H, I, N = 256, 6, 4096, 1024, 4096


def timed_us(fn: Callable[[], object], warmup: int, iterations: int) -> float:
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


def abba(
    fn_a: Callable[[], object],
    fn_b: Callable[[], object],
    warmup: int,
    iterations: int,
    rounds: int,
) -> tuple[list[float], list[float]]:
    samples_a: list[float] = []
    samples_b: list[float] = []
    for _ in range(rounds):
        samples_a.append(timed_us(fn_a, warmup, iterations))
        samples_b.append(timed_us(fn_b, warmup, iterations))
        samples_b.append(timed_us(fn_b, warmup, iterations))
        samples_a.append(timed_us(fn_a, warmup, iterations))
    return samples_a, samples_b


def require_equal(name: str, lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    if torch.equal(lhs, rhs):
        return
    max_abs = (lhs.float() - rhs.float()).abs().max().item()
    raise RuntimeError(f"{name} differs: max_abs={max_abs}")


def capture(fn: Callable[[], tuple[torch.Tensor, ...]]):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = fn()
    return graph, outputs


def run_shape(args: argparse.Namespace, m: int) -> None:
    device = torch.device("cuda")
    cpu_generator = torch.Generator(device="cpu").manual_seed(args.seed + m)

    # These are the raw production tensor layouts, not AIter-preshuffled views.
    w13 = torch.randint(
        0,
        256,
        (E, 2 * I, H // 2),
        dtype=torch.uint8,
        generator=cpu_generator,
    ).to(device)
    s13 = torch.randint(
        110,
        114,
        (E * 2 * I, H // 32),
        dtype=torch.uint8,
        generator=cpu_generator,
    ).to(device)
    w2 = torch.randint(
        0,
        256,
        (E, N, I // 2),
        dtype=torch.uint8,
        generator=cpu_generator,
    ).to(device)
    s2 = torch.randint(
        110,
        114,
        (E * N, I // 32),
        dtype=torch.uint8,
        generator=cpu_generator,
    ).to(device)

    # Unique top-k per token matches the router contract.  Metadata remains
    # static so both captured arms see identical graph-stable sorter buffers.
    topk_ids = torch.rand((m, E), device=device).topk(T, dim=1).indices.to(
        torch.int32
    )
    topk_weights = torch.rand((m, T), dtype=torch.float32, device=device)
    sorted_ids, _, sorted_experts, valid, _ = moe_sorting(
        topk_ids, topk_weights, E, H, torch.bfloat16, block_size=64
    )
    x = torch.randn((m, H), dtype=torch.bfloat16, device=device)
    xq, xs = gfx90a_int8_group32_quant(x)

    def arm_a_debug() -> tuple[torch.Tensor, ...]:
        mid = gfx90a_fp4_expert_gate_up_mfma32(
            xq,
            xs,
            w13,
            s13,
            sorted_ids,
            sorted_experts,
            valid,
            T,
            10.0,
            blocks=args.gate_blocks,
            split=4,
            broadcast_scales=1,
            assignments=64,
            preshuffled=False,
        )
        iq, isc = gfx90a_int8_group32_quant(mid)
        out = gfx90a_fp4_expert_down_mfma32(
            iq,
            isc,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            blocks=args.down_blocks,
            split=2,
            broadcast_scales=1,
            assignments=64,
            preshuffled=False,
        )
        return mid, iq, isc, out

    def arm_b_debug() -> tuple[torch.Tensor, ...]:
        mid, iq, isc = gfx90a_fp4_mfma64_gate_quant_oracle(
            xq,
            xs,
            w13,
            s13,
            sorted_ids,
            sorted_experts,
            valid,
            topk=T,
            blocks=args.gate_blocks,
            split=4,
            debug_intermediate=True,
        )
        assert mid is not None
        out = gfx90a_fp4_expert_down_mfma32(
            iq,
            isc,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            blocks=args.down_blocks,
            split=2,
            broadcast_scales=1,
            assignments=64,
            preshuffled=False,
        )
        return mid, iq, isc, out

    def arm_a_timing() -> tuple[torch.Tensor, ...]:
        return arm_a_debug()

    def arm_b_timing() -> tuple[torch.Tensor, ...]:
        _, iq, isc = gfx90a_fp4_mfma64_gate_quant_oracle(
            xq,
            xs,
            w13,
            s13,
            sorted_ids,
            sorted_experts,
            valid,
            topk=T,
            blocks=args.gate_blocks,
            split=4,
            debug_intermediate=False,
        )
        out = gfx90a_fp4_expert_down_mfma32(
            iq,
            isc,
            w2,
            s2,
            sorted_ids,
            sorted_experts,
            valid,
            topk_weights,
            blocks=args.down_blocks,
            split=2,
            broadcast_scales=1,
            assignments=64,
            preshuffled=False,
        )
        return iq, isc, out

    # Mutate graph-stable input buffers.  Weight bytes/scales are touched too,
    # so equality is not an accident of a single codebook/scale realization.
    for mutation in range(args.mutations):
        fresh_x = torch.randn_like(x)
        fresh_q, fresh_s = gfx90a_int8_group32_quant(fresh_x)
        xq.copy_(fresh_q)
        xs.copy_(fresh_s)
        topk_weights.uniform_()
        w13.view(-1)[mutation % w13.numel()] = (mutation * 29 + 7) & 0xFF
        s13.view(-1)[mutation % s13.numel()] = 110 + mutation % 4
        reference = arm_a_debug()
        candidate = arm_b_debug()
        torch.cuda.synchronize()
        for name, lhs, rhs in zip(
            ("mid", "int8", "scale", "full_routed"), reference, candidate
        ):
            require_equal(f"mutation={mutation} {name}", lhs, rhs)
    print(f"CORRECTNESS M={m} mutations={args.mutations} all_exact=True")

    graph_a, outputs_a = capture(arm_a_timing)
    graph_b, outputs_b = capture(arm_b_timing)
    for replay in range(args.graph_replays):
        # Changes are in-place; capture addresses remain stable.
        xs.mul_(1.0 + ((replay % 3) - 1) * 2.0**-12)
        topk_weights.uniform_()
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()
        require_equal(f"graph={replay} int8", outputs_a[1], outputs_b[0])
        require_equal(f"graph={replay} scale", outputs_a[2], outputs_b[1])
        require_equal(f"graph={replay} full_routed", outputs_a[3], outputs_b[2])
    print(f"GRAPH_CORRECTNESS M={m} replays={args.graph_replays} all_exact=True")

    samples_a, samples_b = abba(
        graph_a.replay,
        graph_b.replay,
        args.warmup,
        args.iterations,
        args.rounds,
    )
    median_a = statistics.median(samples_a)
    median_b = statistics.median(samples_b)
    trim_a = statistics.mean(sorted(samples_a)[1:-1])
    trim_b = statistics.mean(sorted(samples_b)[1:-1])
    print(
        f"ABBA M={m} A_full_routed_us={median_a:.3f} "
        f"B_full_routed_us={median_b:.3f} saved_us={median_a-median_b:.3f} "
        f"delta_pct={(median_b / median_a - 1.0) * 100:+.2f} "
        f"A_trimmed_us={trim_a:.3f} B_trimmed_us={trim_b:.3f} "
        f"A_samples={samples_a} B_samples={samples_b}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[2048, 2304])
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--gate-blocks", type=int, default=1040)
    parser.add_argument("--down-blocks", type=int, default=624)
    parser.add_argument("--seed", type=int, default=20260902)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a required, got {arch}")
    for tokens in args.tokens:
        if tokens not in (2048, 2304):
            raise ValueError("oracle is intentionally restricted to M2048/M2304")
        run_shape(args, tokens)


if __name__ == "__main__":
    main()
