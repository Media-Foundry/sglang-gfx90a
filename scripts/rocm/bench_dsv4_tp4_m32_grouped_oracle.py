#!/usr/bin/env python3
"""Sweep the TP4 M32 grouped FP4 routed stage on a recorded diverse route.

This is a production-shape oracle: E256, top-k 6, H4096 and the TP4 expert
intermediate shard I512.  It keeps the quantization and fixed-slot reduction
order constant while varying only the grouped expert kernel geometry.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 512, 4096
WAVES, LDS_LUT = 8, 2


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts).cuda()

    torch.manual_seed(7)
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device="cuda")

    tied_profiles = (
        ("a8_r2_b624_nolds", 8, 2, 624, 624, 0),
        ("a8_r2_b624", 8, 2, 624, 624, LDS_LUT),
        ("a8_r2_b832", 8, 2, 832, 832, LDS_LUT),
        ("a8_r2_b1040", 8, 2, 1040, 1040, LDS_LUT),
        ("a8_r1_b624", 8, 1, 624, 624, LDS_LUT),
        ("a8_r1_b832", 8, 1, 832, 832, LDS_LUT),
        ("a8_r1_b1040", 8, 1, 1040, 1040, LDS_LUT),
        ("a8_r1_b1248", 8, 1, 1248, 1248, LDS_LUT),
        ("a4_r2_b624", 4, 2, 624, 624, LDS_LUT),
        ("a4_r2_b832", 4, 2, 832, 832, LDS_LUT),
        ("a4_r2_g1040_d832", 4, 2, 1040, 832, LDS_LUT),
        ("a4_r2_g1248_d832", 4, 2, 1248, 832, LDS_LUT),
        ("a4_r2_g1560_d832", 4, 2, 1560, 832, LDS_LUT),
        ("a4_r2_g1664_d832", 4, 2, 1664, 832, LDS_LUT),
        ("a4_r2_g1872_d832", 4, 2, 1872, 832, LDS_LUT),
        ("a4_r2_g2080_d832", 4, 2, 2080, 832, LDS_LUT),
        ("a4_r2_g832_d1040", 4, 2, 832, 1040, LDS_LUT),
        ("a4_r2_g832_d1248", 4, 2, 832, 1248, LDS_LUT),
        ("a4_r4_b832", 4, 4, 832, 832, LDS_LUT),
        ("a4_r4_b1040", 4, 4, 1040, 1040, LDS_LUT),
        ("a2_r2_b832", 2, 2, 832, 832, LDS_LUT),
    )
    profiles = tuple(
        (name, assignments, rows, rows, gate_blocks, down_blocks, lds_lut)
        for name, assignments, rows, gate_blocks, down_blocks, lds_lut
        in tied_profiles
    ) + (
        ("a4_gr2_dr1_g2080_d624", 4, 2, 1, 2080, 624, LDS_LUT),
        ("a4_gr2_dr1_g2080_d832", 4, 2, 1, 2080, 832, LDS_LUT),
        ("a4_gr2_dr1_g2080_d1040", 4, 2, 1, 2080, 1040, LDS_LUT),
        ("a4_gr2_dr4_g2080_d624", 4, 2, 4, 2080, 624, LDS_LUT),
        ("a4_gr2_dr4_g2080_d832", 4, 2, 4, 2080, 832, LDS_LUT),
        ("a4_gr2_dr4_g2080_d1040", 4, 2, 4, 2080, 1040, LDS_LUT),
    )
    outputs: dict[str, torch.Tensor] = {}
    timings: dict[str, list[float]] = {name: [] for name, *_ in profiles}

    for (
        name,
        assignments,
        gate_rows,
        down_rows,
        gate_blocks,
        down_blocks,
        lds_lut,
    ) in profiles:
        metadata = make_metadata(topk_ids, assignments=assignments)
        intermediate = torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda")
        partial = torch.empty((M, T, N), dtype=torch.float32, device="cuda")
        output = torch.empty((M, N), dtype=torch.bfloat16, device="cuda")
        gate = _jit_gate_up_grouped(
            E, M, T, I, H, assignments, gate_rows, WAVES, gate_blocks, lds_lut
        )
        down = _jit_down_grouped(
            E, M, T, N, I, assignments, down_rows, WAVES, down_blocks, lds_lut
        )

        def run() -> None:
            gate.run(
                xq, xs, w13, s13, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid, intermediate, 10.0,
            )
            iq, isc = per_token_group_quant_int8(intermediate, 32)
            down.run_partial(
                iq, isc, w2, s2, metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, topk_weights, partial,
            )
            down.reduce(partial, output)

        run()
        torch.cuda.synchronize()
        outputs[name] = output.clone()
        print(
            f"profile={name} scans={metadata.sorted_experts.numel()} "
            f"padded={metadata.sorted_ids.numel()}", flush=True,
        )
        for _ in range(args.rounds):
            timings[name].append(time_us(run, args.warmup, args.iterations))

    reference = outputs["a8_r2_b624_nolds"]
    for name, *_ in profiles:
        diff = (outputs[name].float() - reference.float()).abs()
        values = timings[name]
        print(
            f"RESULT profile={name} samples_us="
            + ",".join(f"{value:.3f}" for value in values)
            + f" median_us={statistics.median(values):.3f} "
            f"exact={bool(torch.equal(outputs[name], reference))} "
            f"max_abs={float(diff.max()):.8g}",
            flush=True,
        )


if __name__ == "__main__":
    main()
