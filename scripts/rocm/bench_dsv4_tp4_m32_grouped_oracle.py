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
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


E, M, T, H, I, N = 256, 32, 6, 4096, 512, 4096
WAVES, LDS_LUT = 8, 2


@cache_once
def _jit_gate_up_grouped_dpp(
    e: int, m: int, t: int, i: int, k: int, assignments: int, rows: int,
    waves: int, blocks: int, prepacked: int,
):
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_up_grouped_dpp_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[(
            "run",
            f"sglang::Gfx90aFp4ExpertGateUpGroupedDppKernel<{args}>::run",
        )],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down_grouped_dpp(
    e: int, m: int, t: int, n: int, k: int, assignments: int, rows: int,
    waves: int, blocks: int, prepacked: int,
):
    args = make_cpp_args(
        e, m, t, n, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_down_grouped_dpp_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownGroupedDppKernel<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownGroupedDppKernel<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3"],
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


def main() -> None:
    global I
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=8)
    parser.add_argument(
        "--intermediate-size", type=int, default=I,
        help="Local expert intermediate width (512=TP4, 2048=TP1)",
    )
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--waves", type=int, default=WAVES)
    parser.add_argument(
        "--profile",
        help="Run only the named grouped-kernel geometry (useful for profilers)",
    )
    parser.add_argument(
        "--dpp-only", action="store_true",
        help="ABBA the isolated shuffle-versus-DPP A4 TP4 kernels",
    )
    args = parser.parse_args()
    I = args.intermediate_size

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
    w13_prepacked = None
    w2_prepacked = None
    if args.profile is not None and "prepacked" in args.profile:
        def decode_nibble(nibble: torch.Tensor) -> torch.Tensor:
            magnitude = (nibble & 7).to(torch.int8)
            value = (
                magnitude + (magnitude > 4) + (magnitude > 5)
                + 3 * (magnitude > 6)
            )
            return torch.where((nibble & 8) != 0, -value, value).to(torch.int8)

        w13_prepacked = torch.stack(
            (decode_nibble(w13 & 15), decode_nibble(w13 >> 4)), dim=-1
        ).reshape(E, 2 * I, H)
        w2_prepacked = torch.stack(
            (decode_nibble(w2 & 15), decode_nibble(w2 >> 4)), dim=-1
        ).reshape(E, N, I)

    if args.dpp_only:
        metadata = make_metadata(topk_ids, assignments=4)
        shuffle_gate = _jit_gate_up_grouped(
            E, M, T, I, H, 4, 2, args.waves, 2080, LDS_LUT
        )
        dpp_gate = _jit_gate_up_grouped_dpp(
            E, M, T, I, H, 4, 2, args.waves, 2080, LDS_LUT
        )
        shuffle_down = _jit_down_grouped(
            E, M, T, N, I, 4, 2, args.waves, 832, LDS_LUT
        )
        dpp_down = _jit_down_grouped_dpp(
            E, M, T, N, I, 4, 2, args.waves, 832, LDS_LUT
        )
        # A: shuffle/shuffle; G: DPP gate only; D: DPP down only; B: both.
        kernel_matrix = {
            "A": (shuffle_gate, shuffle_down),
            "G": (dpp_gate, shuffle_down),
            "D": (shuffle_gate, dpp_down),
            "B": (dpp_gate, dpp_down),
        }
        states = {}
        stages = {}
        for name, (gate, down) in kernel_matrix.items():
            state = {
                "intermediate": torch.empty(
                    (M, T, I), dtype=torch.bfloat16, device="cuda"
                ),
                "partial": torch.empty(
                    (M, T, N), dtype=torch.float32, device="cuda"
                ),
                "output": torch.empty(
                    (M, N), dtype=torch.bfloat16, device="cuda"
                ),
            }

            def gate_stage(gate=gate, state=state) -> None:
                gate.run(
                    xq, xs, w13, s13, metadata.sorted_ids,
                    metadata.sorted_experts, metadata.valid,
                    state["intermediate"], 10.0,
                )

            def quant_stage(state=state) -> None:
                state["iq"], state["isc"] = per_token_group_quant_int8(
                    state["intermediate"], 32
                )

            def down_stage(down=down, state=state) -> None:
                down.run_partial(
                    state["iq"], state["isc"], w2, s2,
                    metadata.sorted_ids, metadata.sorted_experts,
                    metadata.valid, topk_weights, state["partial"],
                )

            def reduce_stage(down=down, state=state) -> None:
                down.reduce(state["partial"], state["output"])

            def full_stage(
                gate_stage=gate_stage, quant_stage=quant_stage,
                down_stage=down_stage, reduce_stage=reduce_stage,
            ) -> None:
                gate_stage()
                quant_stage()
                down_stage()
                reduce_stage()

            states[name] = state
            stages[name] = {
                "gate": gate_stage,
                "quant": quant_stage,
                "down": down_stage,
                "reduce": reduce_stage,
                "full": full_stage,
            }

        def assert_state_exact(candidate: str, label: str) -> None:
            for tensor_name in ("intermediate", "partial", "output"):
                actual = states[candidate][tensor_name]
                expected = states["A"][tensor_name]
                if not torch.equal(actual, expected):
                    diff = (actual.float() - expected.float()).abs()
                    raise RuntimeError(
                        f"{label} {candidate}.{tensor_name} mismatch: "
                        f"max_abs={float(diff.max())}"
                    )

        for name in ("A", "G", "D", "B"):
            stages[name]["full"]()
        torch.cuda.synchronize()
        for name in ("G", "D", "B"):
            assert_state_exact(name, "initial")

        # Mutate activation and router weights in place so all four paths see
        # identical addresses and values. Check every boundary, not only BF16.
        mutation_input = torch.empty_like(x)
        for mutation in range(100):
            mutation_input.normal_()
            mutation_xq, mutation_xs = per_token_group_quant_int8(
                mutation_input, 32
            )
            xq.copy_(mutation_xq)
            xs.copy_(mutation_xs)
            topk_weights.uniform_()
            for name in ("A", "G", "D", "B"):
                stages[name]["full"]()
            torch.cuda.synchronize()
            for name in ("G", "D", "B"):
                assert_state_exact(name, f"mutation={mutation}")
        print(
            "CORRECTNESS mutations=100 matrix=A/G/D/B "
            "intermediate_exact=True partial_exact=True final_exact=True",
            flush=True,
        )

        # Capture G and D independently. A is recomputed once from the same
        # fixed inputs; replay must preserve every visible boundary 1000 times.
        stages["A"]["full"]()
        torch.cuda.synchronize()
        for candidate in ("G", "D"):
            capture_stream = torch.cuda.Stream()
            capture_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(capture_stream):
                for _ in range(3):
                    stages[candidate]["full"]()
            torch.cuda.current_stream().wait_stream(capture_stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                stages[candidate]["full"]()
            for replay in range(1000):
                graph.replay()
                torch.cuda.synchronize()
                assert_state_exact(candidate, f"graph_replay={replay}")
            print(
                f"CORRECTNESS profile={candidate} graph_replays=1000 "
                "intermediate_exact=True partial_exact=True final_exact=True",
                flush=True,
            )

        # Prepare all stage inputs before timing isolated consumers.
        for name in ("A", "G", "D", "B"):
            stages[name]["full"]()
        torch.cuda.synchronize()
        comparisons = (("G", "A/G/G/A"), ("D", "A/D/D/A"),
                       ("B", "A/B/B/A"))
        for candidate, order_label in comparisons:
            for stage_name in ("gate", "quant", "down", "reduce", "full"):
                timings = {"A": [], candidate: []}
                for _ in range(7):
                    for name in ("A", candidate, candidate, "A"):
                        timings[name].append(time_us(
                            stages[name][stage_name], args.warmup,
                            args.iterations,
                        ))
                for name in ("A", candidate):
                    values = timings[name]
                    trimmed = sorted(values)[1:-1]
                    print(
                        f"RESULT comparison={order_label} stage={stage_name} "
                        f"profile={name} samples_us="
                        + ",".join(f"{value:.3f}" for value in values)
                        + f" median_us={statistics.median(values):.3f}"
                        + f" trimmed_mean_us={statistics.mean(trimmed):.3f}",
                        flush=True,
                    )
        return

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
        ("a4_r2_g2080_d832_nolds", 4, 2, 2080, 832, 0),
        ("a4_r2_g2080_d832_prepacked", 4, 2, 2080, 832, 1),
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
        ("a4_gr1_dr2_g2080_d832", 4, 1, 2, 2080, 832, LDS_LUT),
        ("a4_gr1_dr2_g1664_d1664", 4, 1, 2, 1664, 1664, LDS_LUT),
        ("a4_gr1_dr2_g2080_d1664", 4, 1, 2, 2080, 1664, LDS_LUT),
        ("a4_gr1_dr2_g3120_d832", 4, 1, 2, 3120, 832, LDS_LUT),
        ("a4_gr1_dr2_g4160_d832", 4, 1, 2, 4160, 832, LDS_LUT),
        ("a4_gr2_dr1_g2080_d624", 4, 2, 1, 2080, 624, LDS_LUT),
        ("a4_gr2_dr1_g2080_d832", 4, 2, 1, 2080, 832, LDS_LUT),
        ("a4_gr2_dr1_g2080_d1040", 4, 2, 1, 2080, 1040, LDS_LUT),
        ("a4_gr2_dr4_g2080_d624", 4, 2, 4, 2080, 624, LDS_LUT),
        ("a4_gr2_dr4_g2080_d832", 4, 2, 4, 2080, 832, LDS_LUT),
        ("a4_gr2_dr4_g2080_d1040", 4, 2, 4, 2080, 1040, LDS_LUT),
    )
    if args.profile is not None:
        selected_profiles = set(args.profile.split(","))
        profiles = tuple(
            profile for profile in profiles if profile[0] in selected_profiles
        )
        if not profiles:
            raise ValueError(f"unknown profile: {args.profile}")
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
            E, M, T, I, H, assignments, gate_rows, args.waves, gate_blocks, lds_lut
        )
        down = _jit_down_grouped(
            E, M, T, N, I, assignments, down_rows, args.waves, down_blocks, lds_lut
        )

        def run() -> None:
            gate_weight = w13_prepacked if lds_lut == 1 else w13
            down_weight = w2_prepacked if lds_lut == 1 else w2
            gate.run(
                xq, xs, gate_weight, s13, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid, intermediate, 10.0,
            )
            iq, isc = per_token_group_quant_int8(intermediate, 32)
            down.run_partial(
                iq, isc, down_weight, s2, metadata.sorted_ids, metadata.sorted_experts,
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

    reference = outputs.get(
        "a8_r2_b624_nolds",
        outputs.get("a4_r2_g2080_d832", next(iter(outputs.values()))),
    )
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
