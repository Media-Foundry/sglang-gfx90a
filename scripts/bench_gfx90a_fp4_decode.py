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

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down,
    gfx90a_fp4_expert_down_grouped,
    gfx90a_fp4_expert_down_mfma32,
    gfx90a_fp4_expert_gate_up_grouped,
    gfx90a_fp4_expert_gate_up_mfma32,
)
from sglang.kernels.ops.quantization.gfx90a_int8_quant import (
    gfx90a_int8_group32_quant,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E = 256
TOPK = 6
HIDDEN = 4096
INTERMEDIATE = 256


def make_sorted_metadata(topk_ids: torch.Tensor, assignments: int):
    """Build AIter-compatible expert blocks without importing its JIT stack."""
    ids = topk_ids.cpu().tolist()
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(ids):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    sentinel = topk_ids.shape[0]
    sorted_ids: list[int] = []
    sorted_experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), assignments):
            block = bucket[offset : offset + assignments]
            sorted_ids.extend(block)
            sorted_ids.extend([sentinel] * (assignments - len(block)))
            sorted_experts.append(expert)
    device = topk_ids.device
    return (
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(sorted_experts, dtype=torch.int32, device=device),
        torch.tensor([len(sorted_ids), 0], dtype=torch.int32, device=device),
    )


def unpack_fp4_i8(weight: torch.Tensor) -> torch.Tensor:
    """Offline E2M1 codebook expansion used by the prepacked-weight A/B."""
    lut = torch.tensor(
        [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
        dtype=torch.int8,
        device=weight.device,
    )
    out = torch.empty((*weight.shape[:-1], weight.shape[-1] * 2),
                      dtype=torch.int8, device=weight.device)
    out[..., 0::2] = lut[(weight & 15).long()]
    out[..., 1::2] = lut[(weight >> 4).long()]
    return out


def reconstruct_topk_from_counts(counts: torch.Tensor) -> torch.Tensor:
    """Build a deterministic duplicate-free [32,6] routing with exact counts."""
    counts = counts.to(torch.int64).cpu()
    if counts.shape != (E,) or counts.sum().item() != 32 * TOPK:
        raise ValueError(f"expected 256 counts summing to 192, got {counts.shape}")
    rows: list[list[int]] = [[] for _ in range(32)]
    # Place high-occupancy experts first into the currently least-filled rows.
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert].item())):
            choices = [
                token
                for token in range(32)
                if len(rows[token]) < TOPK and expert not in rows[token]
            ]
            if not choices:
                raise RuntimeError(f"cannot place expert {expert} without duplicates")
            token = min(choices, key=lambda value: (len(rows[value]), value))
            rows[token].append(expert)
    if any(len(row) != TOPK for row in rows):
        raise RuntimeError(f"unbalanced reconstruction: {[len(row) for row in rows]}")
    return torch.tensor(rows, dtype=torch.int32)


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
    parser.add_argument("--prepacked", action="store_true")
    parser.add_argument("--lds-lut", action="store_true")
    parser.add_argument(
        "--compare-direct-down",
        action="store_true",
        help="Also time the token-major direct-final down kernel.",
    )
    parser.add_argument("--recorder", help="Expert recorder .pt for real routing")
    parser.add_argument("--pass-index", type=int, default=47)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--compare-mfma32", action="store_true")
    parser.add_argument("--mfma-blocks", type=int, nargs="+", default=[104, 208, 416])
    parser.add_argument("--mfma-split", type=int, nargs="+", default=[2, 4])
    args = parser.parse_args()
    if args.prepacked and args.lds_lut:
        parser.error("--prepacked and --lds-lut are mutually exclusive")

    if not torch.version.hip:
        raise RuntimeError("this benchmark requires ROCm")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"this benchmark requires gfx90a, got {arch}")

    torch.manual_seed(7)
    device = torch.device("cuda")
    x = torch.randn((args.m, HIDDEN), dtype=torch.bfloat16, device=device)
    if args.recorder:
        if args.m != 32:
            parser.error("--recorder currently requires --m 32")
        payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
        counts = payload["logical_count"][args.pass_index, args.layer] // 8
        topk_ids = reconstruct_topk_from_counts(counts).to(device)
        print(
            f"routing=recorder pass={args.pass_index} layer={args.layer} "
            f"active={(counts > 0).sum().item()} max_occ={counts.max().item()}",
            flush=True,
        )
    else:
        topk_ids = torch.randint(
            0, E, (args.m, TOPK), dtype=torch.int32, device=device
        )
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
    w13_prepacked = unpack_fp4_i8(w13) if args.prepacked else None
    w2_prepacked = unpack_fp4_i8(w2) if args.prepacked else None
    xq, xs = per_token_group_quant_int8(x, 32)
    sorted_ids, sorted_experts, valid = make_sorted_metadata(
        topk_ids, args.assignments
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
                prepacked_weight=w13_prepacked,
                use_lds_lut=args.lds_lut,
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
                prepacked_weight=w2_prepacked,
                use_lds_lut=args.lds_lut,
            )

        median, samples, output = timed_us(
            run, args.warmup, args.iterations, args.rounds
        )
        if args.prepacked or args.lds_lut:
            baseline_intermediate = gfx90a_fp4_expert_gate_up_grouped(
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
            baseline_iq, baseline_isc = per_token_group_quant_int8(
                baseline_intermediate, 32
            )
            baseline_output = gfx90a_fp4_expert_down_grouped(
                baseline_iq,
                baseline_isc,
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
            if not torch.equal(baseline_output, output):
                diff = (baseline_output.float() - output.float()).abs().max()
                raise AssertionError(f"candidate output mismatch: {diff.item()}")
        if reference is None:
            reference = output.clone()
        exact = torch.equal(reference, output)
        print(
            f"m={args.m} assignments={args.assignments} "
            f"gate_rows={rows} down_rows={down_rows} "
            f"prepacked={args.prepacked} lds_lut={args.lds_lut} "
            f"waves={waves} gate_blocks={blocks} "
            f"down_blocks={down_blocks} median_us={median:.3f} "
            f"samples={[round(x, 3) for x in samples]} exact={exact}",
            flush=True,
        )

        if args.compare_direct_down:
            def run_direct_down():
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
                    prepacked_weight=w13_prepacked,
                    use_lds_lut=args.lds_lut,
                )
                iq, isc = per_token_group_quant_int8(intermediate, 32)
                return gfx90a_fp4_expert_down(
                    intermediate,
                    w2,
                    s2,
                    topk_ids,
                    expert_mask=None,
                    topk_weights=topk_weights,
                    live_count=None,
                    prequant=(iq, isc),
                    blocks=down_blocks,
                    rows=down_rows,
                    waves=waves,
                )

            direct_median, direct_samples, direct_output = timed_us(
                run_direct_down, args.warmup, args.iterations, args.rounds
            )
            direct_exact = torch.equal(output, direct_output)
            direct_diff = (
                output.float() - direct_output.float()
            ).abs().max().item()
            print(
                f"direct_down median_us={direct_median:.3f} "
                f"samples={[round(x, 3) for x in direct_samples]} "
                f"exact={direct_exact} max_abs={direct_diff:.6g} "
                f"delta_pct={(direct_median / median - 1.0) * 100.0:+.2f}",
                flush=True,
            )

        if args.compare_mfma32:
            mfma_sorted_ids, mfma_sorted_experts, mfma_valid = (
                make_sorted_metadata(topk_ids, 32)
            )
            for mfma_blocks, mfma_split in itertools.product(
                args.mfma_blocks, args.mfma_split
            ):
                def run_mfma():
                    intermediate = gfx90a_fp4_expert_gate_up_mfma32(
                        xq,
                        xs,
                        w13,
                        s13,
                        mfma_sorted_ids,
                        mfma_sorted_experts,
                        mfma_valid,
                        TOPK,
                        10.0,
                        blocks=mfma_blocks,
                        split=mfma_split,
                        assignments=32,
                    )
                    iq, isc = gfx90a_int8_group32_quant(intermediate)
                    return gfx90a_fp4_expert_down_mfma32(
                        iq,
                        isc,
                        w2,
                        s2,
                        mfma_sorted_ids,
                        mfma_sorted_experts,
                        mfma_valid,
                        topk_weights,
                        blocks=mfma_blocks,
                        split=mfma_split,
                        assignments=32,
                    )

                mfma_median, mfma_samples, mfma_output = timed_us(
                    run_mfma, args.warmup, args.iterations, args.rounds
                )
                mfma_exact = torch.equal(output, mfma_output)
                mfma_diff = (
                    output.float() - mfma_output.float()
                ).abs().max().item()
                rel_l2 = (
                    torch.linalg.vector_norm(
                        output.float() - mfma_output.float()
                    )
                    / torch.linalg.vector_norm(output.float()).clamp_min(1e-12)
                ).item()
                print(
                    f"mfma32 blocks={mfma_blocks} split={mfma_split} "
                    f"median_us={mfma_median:.3f} "
                    f"samples={[round(x, 3) for x in mfma_samples]} "
                    f"exact={mfma_exact} max_abs={mfma_diff:.6g} "
                    f"rel_l2={rel_l2:.6g} "
                    f"delta_pct={(mfma_median / median - 1.0) * 100.0:+.2f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
