#!/usr/bin/env python3
"""ABBA the TP4 M32 projection-specialized paired-wave gate/up oracle."""

from __future__ import annotations

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 512, 4096
A, R, W, G, D, LUT = 4, 2, 8, 2080, 832, 2


@cache_once
def _jit_paired(blocks: int):
    args = make_cpp_args(E, M, T, I, H, A, R, blocks, LUT)
    return load_jit(
        "gfx90a_fp4_expert_gate_up_paired_projection_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gate_up_paired_oracle.cuh"
        ],
        cuda_wrappers=[(
            "run",
            f"sglang::Gfx90aFp4ExpertGateUpPairedOracleKernel<{args}>::run",
        )],
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


def trimmed(values: list[float]) -> float:
    return statistics.mean(sorted(values)[1:-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", required=True)
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    args = parser.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts).cuda()
    metadata = make_metadata(topk_ids, assignments=A)
    scans = metadata.sorted_experts.numel()
    paired_blocks = (scans * (I // R) + 7) // 8
    print(
        f"ROUTE active={int((counts > 0).sum())} assignments={int(counts.sum())} "
        f"scans={scans} paired_blocks={paired_blocks}",
        flush=True,
    )

    torch.manual_seed(20260830)
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device="cuda")

    reference_gate = _jit_gate_up_grouped(E, M, T, I, H, A, R, W, G, LUT)
    paired_gate = _jit_paired(paired_blocks)
    down = _jit_down_grouped(E, M, T, N, I, A, R, W, D, LUT)

    states = {}
    stages = {}
    for name, gate in (("A", reference_gate), ("B", paired_gate)):
        state = {
            "intermediate": torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda"),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device="cuda"),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        }

        def gate_stage(gate=gate, state=state):
            gate.run(
                xq, xs, w13, s13, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid,
                state["intermediate"], 10.0,
            )

        def quant_stage(state=state):
            state["iq"], state["isc"] = per_token_group_quant_int8(
                state["intermediate"], 32
            )

        def down_stage(state=state):
            down.run_partial(
                state["iq"], state["isc"], w2, s2,
                metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, topk_weights, state["partial"],
            )

        def reduce_stage(state=state):
            down.reduce(state["partial"], state["output"])

        def full_stage(
            gate_stage=gate_stage, quant_stage=quant_stage,
            down_stage=down_stage, reduce_stage=reduce_stage,
        ):
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

    def assert_exact(label: str) -> None:
        for tensor_name in ("intermediate", "iq", "isc", "partial", "output"):
            expected = states["A"][tensor_name]
            actual = states["B"][tensor_name]
            if not torch.equal(expected, actual):
                diff = (expected.float() - actual.float()).abs()
                raise RuntimeError(
                    f"{label} {tensor_name} mismatch max_abs={float(diff.max())}"
                )

    for name in ("A", "B"):
        stages[name]["full"]()
    torch.cuda.synchronize()
    assert_exact("initial")

    mutation_input = torch.empty_like(x)
    for mutation in range(args.mutations):
        mutation_input.normal_()
        mxq, mxs = per_token_group_quant_int8(mutation_input, 32)
        xq.copy_(mxq)
        xs.copy_(mxs)
        topk_weights.uniform_()
        stages["A"]["full"]()
        stages["B"]["full"]()
        torch.cuda.synchronize()
        assert_exact(f"mutation={mutation}")
    print(
        f"CORRECTNESS mutations={args.mutations} intermediate_exact=True "
        "quant_exact=True partial_exact=True final_exact=True",
        flush=True,
    )

    # Capture both paths independently, then mutate graph inputs and replay.
    graphs = {}
    for name in ("A", "B"):
        capture_stream = torch.cuda.Stream()
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            for _ in range(3):
                stages[name]["full"]()
        torch.cuda.current_stream().wait_stream(capture_stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stages[name]["full"]()
        graphs[name] = graph
    for replay in range(args.mutations):
        mutation_input.normal_()
        mxq, mxs = per_token_group_quant_int8(mutation_input, 32)
        xq.copy_(mxq)
        xs.copy_(mxs)
        topk_weights.uniform_()
        graphs["A"].replay()
        graphs["B"].replay()
        torch.cuda.synchronize()
        assert_exact(f"graph_replay={replay}")
    print(
        f"CORRECTNESS graph_mutation_replays={args.mutations} "
        "intermediate_exact=True quant_exact=True partial_exact=True final_exact=True",
        flush=True,
    )

    # Restore fixed inputs and ABBA every visible stage.
    mutation_input.normal_()
    mxq, mxs = per_token_group_quant_int8(mutation_input, 32)
    xq.copy_(mxq)
    xs.copy_(mxs)
    timings = {
        stage_name: {"A": [], "B": []}
        for stage_name in ("gate", "quant", "down", "reduce", "full")
    }
    for _ in range(args.rounds):
        for name in ("A", "B", "B", "A"):
            stages[name]["full"]()
            for stage_name in timings:
                timings[stage_name][name].append(
                    time_us(stages[name][stage_name], args.warmup, args.iterations)
                )
    for stage_name, profiles in timings.items():
        for name, values in profiles.items():
            print(
                f"RESULT stage={stage_name} profile={name} samples_us="
                + ",".join(f"{value:.3f}" for value in values)
                + f" median_us={statistics.median(values):.3f} "
                f"trimmed_mean_us={trimmed(values):.3f}",
                flush=True,
            )
    baseline = trimmed(timings["full"]["A"])
    candidate = trimmed(timings["full"]["B"])
    gain = (baseline / candidate - 1.0) * 100.0
    print(
        f"DECISION baseline_us={baseline:.3f} candidate_us={candidate:.3f} "
        f"gain_pct={gain:.3f} passes_10pct={candidate <= 0.9 * baseline} "
        f"passes_395us={candidate <= 395.0}",
        flush=True,
    )


if __name__ == "__main__":
    main()
