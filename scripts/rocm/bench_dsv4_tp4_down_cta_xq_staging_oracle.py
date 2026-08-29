#!/usr/bin/env python3
"""Standalone ABBA for oracle-only CTA-wide down activation LDS staging."""

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
    _jit_down_grouped_row_prefetch,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 512, 4096
A, R, W, G, D, LDS = 4, 2, 8, 2080, 832, 2


@cache_once
def candidate_module():
    args = make_cpp_args(E, M, T, N, I, A, W, D, LDS)
    return load_jit(
        "gfx90a_fp4_expert_down_cta_xq_staging_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_down_cta_xq_staging_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownCtaXqStagingOracle<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownCtaXqStagingOracle<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3", "-Rpass-analysis=kernel-resource-usage"],
    )


def time_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1787803355.1855972.pt",
    )
    parser.add_argument("--pass-index", type=int, default=37)
    parser.add_argument("--layer", type=int, default=34)
    parser.add_argument("--recorded-world-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--compile-only", action="store_true")
    args = parser.parse_args()

    candidate = candidate_module()
    if args.compile_only:
        print("candidate_compile=ok")
        return

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.recorded_world_size) != 0):
        raise RuntimeError("recorded counts are not divisible by world size")
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts).cuda()
    metadata = make_metadata(topk_ids, assignments=A)

    torch.manual_seed(7)
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device="cuda")

    gate = _jit_gate_up_grouped(E, M, T, I, H, A, R, W, G, LDS)
    baseline = _jit_down_grouped_row_prefetch(E, M, T, N, I, A, W, D, LDS)
    states, stages = {}, {}
    for name, down in (("A", baseline), ("B", candidate)):
        state = {
            "intermediate": torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda"),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device="cuda"),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        }

        def gate_stage(state=state):
            gate.run(xq, xs, w13, s13, metadata.sorted_ids,
                     metadata.sorted_experts, metadata.valid,
                     state["intermediate"], 10.0)

        def quant_stage(state=state):
            state["iq"], state["isc"] = per_token_group_quant_int8(
                state["intermediate"], 32
            )

        def down_stage(state=state, down=down):
            down.run_partial(state["iq"], state["isc"], w2, s2,
                             metadata.sorted_ids, metadata.sorted_experts,
                             metadata.valid, topk_weights, state["partial"])

        def reduce_stage(state=state, down=down):
            down.reduce(state["partial"], state["output"])

        def full_stage(gs=gate_stage, qs=quant_stage, ds=down_stage, rs=reduce_stage):
            gs(); qs(); ds(); rs()

        states[name] = state
        stages[name] = {"down": down_stage, "full": full_stage}
        full_stage()
    torch.cuda.synchronize()

    def assert_exact(label: str) -> None:
        for tensor_name in ("intermediate", "partial", "output"):
            lhs, rhs = states["A"][tensor_name], states["B"][tensor_name]
            if not torch.equal(lhs, rhs):
                diff = (lhs.float() - rhs.float()).abs().max().item()
                raise RuntimeError(f"{label} {tensor_name} mismatch max_abs={diff}")

    assert_exact("initial")
    mutation_input = torch.empty_like(x)
    for mutation in range(100):
        mutation_input.normal_()
        q, scale = per_token_group_quant_int8(mutation_input, 32)
        xq.copy_(q); xs.copy_(scale); topk_weights.uniform_()
        stages["A"]["full"](); stages["B"]["full"]()
        torch.cuda.synchronize()
        assert_exact(f"mutation={mutation}")
    print("CORRECTNESS mutations=100 intermediate_exact=True partial_exact=True final_exact=True")

    for stage_name in ("down", "full"):
        timings = {"A": [], "B": []}
        for _ in range(7):
            for name in ("A", "B", "B", "A"):
                timings[name].append(
                    time_us(stages[name][stage_name], args.warmup, args.iterations)
                )
        for name in ("A", "B"):
            values = timings[name]
            trimmed = sorted(values)[1:-1]
            print(
                f"RESULT stage={stage_name} profile={name} samples_us="
                + ",".join(f"{value:.3f}" for value in values)
                + f" median_us={statistics.median(values):.3f}"
                + f" trimmed_mean_us={statistics.mean(trimmed):.3f}"
            )


if __name__ == "__main__":
    main()
