#!/usr/bin/env python3
"""Graph ABBA for two 512-thread projection-only grouped FP4 kernels."""

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
def _jit_dual_projection():
    args = make_cpp_args(E, M, T, I, H, A, R, W, G, LUT)
    return load_jit(
        "gfx90a_fp4_expert_gate_up_dual_stream_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gate_up_dual_stream_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "gate",
                "sglang::Gfx90aFp4ExpertProjectionGroupedOracleKernel<"
                f"{args}, false>::run",
            ),
            (
                "up",
                "sglang::Gfx90aFp4ExpertProjectionGroupedOracleKernel<"
                f"{args}, true>::run",
            ),
            (
                "combine",
                f"sglang::Gfx90aFp4ExpertProjectionCombineOracleKernel<"
                f"{M}, {T}, {I}, 384>::run",
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
    counts = raw // args.recorded_world_size
    topk_ids = reconstruct_topk_from_counts(counts).cuda()
    metadata = make_metadata(topk_ids, assignments=A)
    print(
        f"ROUTE active={int((counts > 0).sum())} assignments={int(counts.sum())} "
        f"scans={metadata.sorted_experts.numel()}",
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
    dual = _jit_dual_projection()
    down = _jit_down_grouped(E, M, T, N, I, A, R, W, D, LUT)
    aux = torch.cuda.Stream()
    fork = torch.cuda.Event()
    up_done = torch.cuda.Event()

    states = {
        name: {
            "intermediate": torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda"),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device="cuda"),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        }
        for name in ("A", "B")
    }
    states["B"]["gate_fp32"] = torch.empty((M, T, I), dtype=torch.float32, device="cuda")
    states["B"]["up_fp32"] = torch.empty((M, T, I), dtype=torch.float32, device="cuda")

    def reference_producer():
        reference_gate.run(
            xq, xs, w13, s13, metadata.sorted_ids, metadata.sorted_experts,
            metadata.valid, states["A"]["intermediate"], 10.0,
        )

    def candidate_producer():
        current = torch.cuda.current_stream()
        fork.record(current)
        aux.wait_event(fork)
        dual.gate(
            xq, xs, w13, s13, metadata.sorted_ids, metadata.sorted_experts,
            metadata.valid, states["B"]["gate_fp32"],
        )
        with torch.cuda.stream(aux):
            dual.up(
                xq, xs, w13, s13, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid,
                states["B"]["up_fp32"],
            )
            up_done.record(aux)
        current.wait_event(up_done)
        dual.combine(
            states["B"]["gate_fp32"], states["B"]["up_fp32"],
            states["B"]["intermediate"], 10.0,
        )

    producers = {"A": reference_producer, "B": candidate_producer}
    stages = {}
    for name in ("A", "B"):
        state = states[name]

        def quant(state=state):
            state["iq"], state["isc"] = per_token_group_quant_int8(
                state["intermediate"], 32
            )

        def down_stage(state=state):
            down.run_partial(
                state["iq"], state["isc"], w2, s2,
                metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, topk_weights, state["partial"],
            )

        def reduce(state=state):
            down.reduce(state["partial"], state["output"])

        def full(producer=producers[name], quant=quant,
                 down_stage=down_stage, reduce=reduce):
            producer()
            quant()
            down_stage()
            reduce()

        stages[name] = {
            "producer": producers[name], "quant": quant, "down": down_stage,
            "reduce": reduce, "full": full,
        }

    def assert_exact(label: str):
        for tensor_name in ("intermediate", "iq", "isc", "partial", "output"):
            a, b = states["A"][tensor_name], states["B"][tensor_name]
            if not torch.equal(a, b):
                diff = (a.float() - b.float()).abs()
                raise RuntimeError(
                    f"{label} {tensor_name} mismatch max_abs={float(diff.max())}"
                )

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
    print(f"CORRECTNESS mutations={args.mutations} all_boundaries_exact=True", flush=True)

    graphs = {}
    for name in ("A", "B"):
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
    print(f"CORRECTNESS graph_mutation_replays={args.mutations} all_boundaries_exact=True", flush=True)

    timings = {
        stage: {"A": [], "B": []}
        for stage in ("producer", "quant", "down", "reduce", "full", "graph")
    }
    for _ in range(args.rounds):
        for name in ("A", "B", "B", "A"):
            stages[name]["full"]()
            for stage in ("producer", "quant", "down", "reduce", "full"):
                timings[stage][name].append(
                    time_us(stages[name][stage], args.warmup, args.iterations)
                )
            timings["graph"][name].append(
                time_us(graphs[name].replay, args.warmup, args.iterations)
            )
    for stage, profiles in timings.items():
        for name, values in profiles.items():
            print(
                f"RESULT stage={stage} profile={name} median_us="
                f"{statistics.median(values):.3f} trimmed_mean_us={trimmed(values):.3f}",
                flush=True,
            )
    baseline = trimmed(timings["graph"]["A"])
    candidate = trimmed(timings["graph"]["B"])
    print(
        f"DECISION graph_baseline_us={baseline:.3f} graph_candidate_us={candidate:.3f} "
        f"gain_pct={(baseline / candidate - 1) * 100:.3f} "
        f"passes_395us={candidate <= 395.0} passes_10pct={candidate <= baseline * 0.9}",
        flush=True,
    )


if __name__ == "__main__":
    main()
