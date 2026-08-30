#!/usr/bin/env python3
"""Standalone M32 A1/A2/A4 routed-FP4 occupancy-bucket oracle.

This deliberately does not wire a production selector.  It reconstructs one
duplicate-free M32 routing from a TP8 expert-recorder pass, builds fixed CPU
metadata once, and compares:

  A: one A4 gate launch + quant + one A4 down-partial launch + one reduction
  B: A1/A2/A4 gate launches + quant + A1/A2/A4 down-partial launches + one
     identical fixed-slot reduction

The bucket producer kernels share the same ``[M,T,I]`` gate output and
``[M,T,N]`` FP32 down partial, so B does not pay three reductions.  Metadata
construction is reported separately and is not included in GPU stage timing;
a production GPU sorter would need to generate the same bucket layout.
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import dataclass

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E = 256
M = 32
T = 6
H = 4096
I = 256
N = 4096
ROWS = 2
WAVES = 8
LDS_LUT = 2


@dataclass(frozen=True)
class Metadata:
    assignments: int
    sorted_ids: torch.Tensor
    sorted_experts: torch.Tensor
    valid: torch.Tensor


@dataclass(frozen=True)
class BlockProfile:
    name: str
    gate: tuple[int, int, int]
    down: tuple[int, int, int]


PROFILES = (
    BlockProfile("b832_416_416", (832, 416, 416), (832, 416, 416)),
    BlockProfile("b624_208_208_d832_416_416", (624, 208, 208), (832, 416, 416)),
    BlockProfile("b416_208_208_d832_416_416", (416, 208, 208), (832, 416, 416)),
    BlockProfile("b624_208_208", (624, 208, 208), (624, 208, 208)),
    BlockProfile("b416_104_104_d624_208_208", (416, 104, 104), (624, 208, 208)),
)


def reconstruct_topk_from_counts(
    counts: torch.Tensor, m: int = M, topk: int = T
) -> torch.Tensor:
    """Deterministically place exact counts into duplicate-free rows."""
    counts = counts.to(torch.int64).cpu()
    if counts.shape != (E,) or counts.sum().item() != m * topk:
        raise ValueError(
            f"expected [256] counts summing to {m * topk}, got "
            f"shape={counts.shape} sum={counts.sum().item()}"
        )
    rows: list[list[int]] = [[] for _ in range(m)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [
                token
                for token in range(m)
                if len(rows[token]) < topk and expert not in rows[token]
            ]
            if not choices:
                raise RuntimeError(f"cannot place expert {expert} without duplicates")
            token = min(choices, key=lambda value: (len(rows[value]), value))
            rows[token].append(expert)
    result = torch.tensor(rows, dtype=torch.int32)
    if result.shape != (m, topk):
        raise RuntimeError(f"bad reconstructed shape {tuple(result.shape)}")
    for row in result.tolist():
        if len(set(row)) != topk:
            raise RuntimeError("duplicate expert in reconstructed top-k row")
    return result


def make_metadata(
    topk_ids: torch.Tensor,
    *,
    assignments: int,
    occupancy: int | None = None,
) -> Metadata:
    """Build AIter-compatible blocks, optionally selecting one occupancy class."""
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    # The padding token must be invalid for the runtime shape.  Using the
    # module's historical M32 constant here aliases a real token at M64 and
    # turns padded assignments into racing writes to token 32.
    sentinel = topk_ids.shape[0]
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        if not bucket:
            continue
        selected = (
            occupancy is None
            or (occupancy == 1 and len(bucket) == 1)
            or (occupancy == 2 and len(bucket) == 2)
            or (occupancy == 4 and len(bucket) >= 3)
            or (occupancy == 0 and len(bucket) >= 2)
        )
        if not selected:
            continue
        for offset in range(0, len(bucket), assignments):
            block = bucket[offset : offset + assignments]
            ids.extend(block)
            ids.extend([sentinel] * (assignments - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return Metadata(
        assignments,
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def invoke_gate(
    metadata: Metadata,
    blocks: int,
    xq: torch.Tensor,
    xs: torch.Tensor,
    w13: torch.Tensor,
    s13: torch.Tensor,
    out: torch.Tensor,
) -> None:
    _jit_gate_up_grouped(
        E, M, T, I, H, metadata.assignments, ROWS, WAVES, blocks, LDS_LUT
    ).run(
        xq,
        xs,
        w13,
        s13,
        metadata.sorted_ids,
        metadata.sorted_experts,
        metadata.valid,
        out,
        10.0,
    )


def invoke_down_partial(
    metadata: Metadata,
    blocks: int,
    iq: torch.Tensor,
    isc: torch.Tensor,
    w2: torch.Tensor,
    s2: torch.Tensor,
    topk_weights: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    _jit_down_grouped(
        E, M, T, N, I, metadata.assignments, ROWS, WAVES, blocks, LDS_LUT
    ).run_partial(
        iq,
        isc,
        w2,
        s2,
        metadata.sorted_ids,
        metadata.sorted_experts,
        metadata.valid,
        topk_weights,
        partial,
    )


def reduce_once(partial: torch.Tensor, out: torch.Tensor) -> None:
    # Reduction is independent of assignment width; reuse the A4/B832 module.
    _jit_down_grouped(E, M, T, N, I, 4, ROWS, WAVES, 832, LDS_LUT).reduce(
        partial, out
    )


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


def abba_pair(fn_a, fn_b, *, warmup: int, iterations: int, rounds: int):
    a_samples: list[float] = []
    b_samples: list[float] = []
    for _ in range(rounds):
        a_samples.append(time_segment(fn_a, warmup=warmup, iterations=iterations))
        b_samples.append(time_segment(fn_b, warmup=warmup, iterations=iterations))
        b_samples.append(time_segment(fn_b, warmup=warmup, iterations=iterations))
        a_samples.append(time_segment(fn_a, warmup=warmup, iterations=iterations))
    return a_samples, b_samples


def main() -> None:
    global I
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    parser.add_argument("--intermediate-size", type=int, default=I)
    parser.add_argument("--baseline-gate-blocks", type=int, default=832)
    parser.add_argument("--baseline-down-blocks", type=int, default=832)
    parser.add_argument("--two-bucket", action="store_true")
    parser.add_argument("--a1-gate-blocks", type=int, default=416)
    parser.add_argument("--rest-gate-blocks", type=int, default=1664)
    parser.add_argument("--a1-down-blocks", type=int, default=416)
    parser.add_argument("--rest-down-blocks", type=int, default=832)
    args = parser.parse_args()
    I = args.intermediate_size
    profiles = PROFILES
    if args.two_bucket:
        profiles = (
            BlockProfile(
                "a1_plus_a4rest",
                (args.a1_gate_blocks, args.rest_gate_blocks),
                (args.a1_down_blocks, args.rest_down_blocks),
            ),
        )

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")

    torch.manual_seed(7)
    device = torch.device("cuda")
    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.world_size) != 0):
        raise RuntimeError("recorder counts are not divisible by TP world size")
    counts = raw // args.world_size
    topk_ids = reconstruct_topk_from_counts(counts).to(device)

    metadata_begin = time.perf_counter_ns()
    a4 = make_metadata(topk_ids, assignments=4)
    bucket_metadata = (
        (
            make_metadata(topk_ids, assignments=1, occupancy=1),
            make_metadata(topk_ids, assignments=4, occupancy=0),
        )
        if args.two_bucket
        else (
            make_metadata(topk_ids, assignments=1, occupancy=1),
            make_metadata(topk_ids, assignments=2, occupancy=2),
            make_metadata(topk_ids, assignments=4, occupancy=4),
        )
    )
    metadata_us = (time.perf_counter_ns() - metadata_begin) / 1000.0
    bucket_names = ("A1", "A4-rest") if args.two_bucket else ("A1", "A2", "A4")
    represented = 0
    for name, metadata in zip(bucket_names, bucket_metadata):
        valid = metadata.sorted_ids.cpu()
        represented += int(((valid & 0x00FFFFFF) < M).sum())
        print(
            f"bucket={name} blocks={metadata.sorted_experts.numel()} "
            f"padded_ids={metadata.sorted_ids.numel()}",
            flush=True,
        )
    if represented != M * T:
        raise RuntimeError(f"bucket metadata represents {represented}, expected {M*T}")
    print(
        f"routing pass={args.pass_index} layer={args.layer} "
        f"active={int((counts > 0).sum())} max_occ={int(counts.max())} "
        f"a4_blocks={a4.sorted_experts.numel()} metadata_cpu_us={metadata_us:.3f}",
        flush=True,
    )

    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)
    intermediate_a = torch.empty((M, T, I), dtype=torch.bfloat16, device=device)
    intermediate_b = torch.empty_like(intermediate_a)
    partial_a = torch.empty((M, T, N), dtype=torch.float32, device=device)
    partial_b = torch.empty_like(partial_a)
    out_a = torch.empty((M, N), dtype=torch.bfloat16, device=device)
    out_b = torch.empty_like(out_a)

    def run_a() -> torch.Tensor:
        invoke_gate(
            a4, args.baseline_gate_blocks, xq, xs, w13, s13, intermediate_a
        )
        iq, isc = per_token_group_quant_int8(intermediate_a, 32)
        invoke_down_partial(
            a4, args.baseline_down_blocks, iq, isc, w2, s2,
            topk_weights, partial_a
        )
        reduce_once(partial_a, out_a)
        return out_a

    def make_b(profile: BlockProfile):
        def run_b() -> torch.Tensor:
            for metadata, blocks in zip(bucket_metadata, profile.gate):
                invoke_gate(metadata, blocks, xq, xs, w13, s13, intermediate_b)
            iq, isc = per_token_group_quant_int8(intermediate_b, 32)
            for metadata, blocks in zip(bucket_metadata, profile.down):
                invoke_down_partial(
                    metadata, blocks, iq, isc, w2, s2, topk_weights, partial_b
                )
            reduce_once(partial_b, out_b)
            return out_b

        return run_b

    # Compile and prove exactness before timing any profile.
    reference = run_a().clone()
    torch.cuda.synchronize()
    for profile in profiles:
        candidate = make_b(profile)()
        torch.cuda.synchronize()
        gate_exact = torch.equal(intermediate_a, intermediate_b)
        partial_exact = torch.equal(partial_a, partial_b)
        output_exact = torch.equal(reference, candidate)
        max_abs = (reference.float() - candidate.float()).abs().max().item()
        print(
            f"correctness profile={profile.name} gate_exact={gate_exact} "
            f"partial_exact={partial_exact} output_exact={output_exact} "
            f"max_abs={max_abs:.8g}",
            flush=True,
        )
        if not (gate_exact and partial_exact and output_exact):
            raise AssertionError(f"{profile.name} is not bitwise exact")

    # Block geometry cannot change arithmetic, so stress the first profile
    # over mutated quantized inputs after every geometry has passed its fixed
    # input check.  This catches stale shared-output/partial slots cheaply.
    stress_b = make_b(profiles[0])
    for replay in range(args.correctness_replays):
        xq.add_((replay % 7) + 1)
        reference = run_a().clone()
        candidate = stress_b()
        torch.cuda.synchronize()
        if not torch.equal(reference, candidate):
            diff = (reference.float() - candidate.float()).abs().max().item()
            raise AssertionError(
                f"mutated correctness replay {replay} mismatch max_abs={diff}"
            )
    print(
        f"correctness_mutations={args.correctness_replays} output_exact=True",
        flush=True,
    )

    for profile in profiles:
        run_b = make_b(profile)
        a_samples, b_samples = abba_pair(
            run_a,
            run_b,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        a_median, b_median = map(statistics.median, (a_samples, b_samples))
        print(
            f"ABBA profile={profile.name} A_median_us={a_median:.3f} "
            f"B_median_us={b_median:.3f} "
            f"delta_pct={(b_median/a_median-1)*100:+.2f} "
            f"gate_blocks={profile.gate} down_blocks={profile.down} "
            f"A_samples={[round(v,3) for v in a_samples]} "
            f"B_samples={[round(v,3) for v in b_samples]}",
            flush=True,
        )

        def gate_a() -> None:
            invoke_gate(
                a4, args.baseline_gate_blocks, xq, xs, w13, s13, intermediate_a
            )

        def gate_b() -> None:
            for metadata, blocks in zip(bucket_metadata, profile.gate):
                invoke_gate(metadata, blocks, xq, xs, w13, s13, intermediate_b)

        # Gate outputs are already proven exact, so one common quantized input
        # isolates the down producer without changing its numerical path.
        gate_a()
        iq_fixed, isc_fixed = per_token_group_quant_int8(intermediate_a, 32)

        def down_a() -> None:
            invoke_down_partial(
                a4, args.baseline_down_blocks, iq_fixed, isc_fixed, w2, s2,
                topk_weights, partial_a
            )

        def down_b() -> None:
            for metadata, blocks in zip(bucket_metadata, profile.down):
                invoke_down_partial(
                    metadata,
                    blocks,
                    iq_fixed,
                    isc_fixed,
                    w2,
                    s2,
                    topk_weights,
                    partial_b,
                )

        gate_a_samples, gate_b_samples = abba_pair(
            gate_a,
            gate_b,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        down_a_samples, down_b_samples = abba_pair(
            down_a,
            down_b,
            warmup=args.warmup,
            iterations=args.iterations,
            rounds=args.rounds,
        )
        quant_samples = [
            time_segment(
                lambda: per_token_group_quant_int8(intermediate_a, 32),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for _ in range(args.rounds)
        ]
        reduce_samples = [
            time_segment(
                lambda: reduce_once(partial_a, out_a),
                warmup=args.warmup,
                iterations=args.iterations,
            )
            for _ in range(args.rounds)
        ]
        print(
            f"components profile={profile.name} "
            f"gate_A_us={statistics.median(gate_a_samples):.3f} "
            f"gate_B_us={statistics.median(gate_b_samples):.3f} "
            f"down_A_us={statistics.median(down_a_samples):.3f} "
            f"down_B_us={statistics.median(down_b_samples):.3f} "
            f"quant_us={statistics.median(quant_samples):.3f} "
            f"reduce_us={statistics.median(reduce_samples):.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
