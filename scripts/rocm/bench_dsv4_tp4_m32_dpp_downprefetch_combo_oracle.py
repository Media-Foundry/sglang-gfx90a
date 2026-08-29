#!/usr/bin/env python3
"""DPP gate wave/grid sweep with grouped-down row prefetch fixed."""

from __future__ import annotations

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from scripts.rocm.bench_dsv4_tp4_down_row_prefetch_oracle import candidate_module
from scripts.rocm.bench_dsv4_tp4_m32_grouped_oracle import (
    _jit_gate_up_grouped_dpp,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, M, T, H, I, N = 256, 32, 6, 4096, 512, 4096
A4, R2, G, D, LUT = 4, 2, 2080, 832, 2


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
    parser.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1787803355.1855972.pt",
    )
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
    metadata = make_metadata(topk_ids, assignments=A4)
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

    gates = {
        "W8G2080": _jit_gate_up_grouped_dpp(
            E, M, T, I, H, A4, R2, 8, G, LUT
        ),
        **{
            f"W4G{blocks}": _jit_gate_up_grouped_dpp(
                E, M, T, I, H, A4, 1, 4, blocks, LUT
            )
            for blocks in (1664, 2080, 2496, 3120)
        },
    }
    prefetched_down = candidate_module()
    profiles = tuple(gates)

    states = {}
    stages = {}
    for name in profiles:
        state = {
            "intermediate": torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda"),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device="cuda"),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device="cuda"),
        }
        gate, down = gates[name], prefetched_down

        def gate_stage(state=state, gate=gate):
            gate.run(
                xq, xs, w13, s13, metadata.sorted_ids,
                metadata.sorted_experts, metadata.valid,
                state["intermediate"], 10.0,
            )

        def quant_stage(state=state):
            state["iq"], state["isc"] = per_token_group_quant_int8(
                state["intermediate"], 32
            )

        def down_stage(state=state, down=down):
            down.run_partial(
                state["iq"], state["isc"], w2, s2,
                metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, topk_weights, state["partial"],
            )

        def reduce_stage(state=state, down=down):
            down.reduce(state["partial"], state["output"])

        def full_stage(
            gate_stage=gate_stage, quant_stage=quant_stage,
            down_stage=down_stage, reduce_stage=reduce_stage,
        ):
            gate_stage(); quant_stage(); down_stage(); reduce_stage()

        states[name] = state
        stages[name] = {
            "gate": gate_stage, "quant": quant_stage, "down": down_stage,
            "reduce": reduce_stage, "full": full_stage,
        }

    def assert_exact(candidate: str, label: str):
        for tensor_name in ("intermediate", "iq", "isc", "partial", "output"):
            expected, actual = (
                states["W8G2080"][tensor_name],
                states[candidate][tensor_name],
            )
            if not torch.equal(expected, actual):
                diff = (expected.float() - actual.float()).abs()
                raise RuntimeError(
                    f"{label} {candidate}.{tensor_name} mismatch "
                    f"max_abs={float(diff.max())}"
                )

    mutation_input = torch.empty_like(x)
    for mutation in range(args.mutations):
        mutation_input.normal_()
        mxq, mxs = per_token_group_quant_int8(mutation_input, 32)
        xq.copy_(mxq); xs.copy_(mxs); topk_weights.uniform_()
        for name in profiles:
            stages[name]["full"]()
        torch.cuda.synchronize()
        for name in profiles[1:]:
            assert_exact(name, f"mutation={mutation}")
    print(
        f"CORRECTNESS mutations={args.mutations} profiles={profiles} "
        "intermediate_exact=True quant_exact=True partial_exact=True final_exact=True",
        flush=True,
    )

    timings = {
        stage_name: {name: [] for name in profiles}
        for stage_name in ("gate", "quant", "down", "reduce", "full")
    }
    for _ in range(args.rounds):
        for name in profiles + tuple(reversed(profiles)):
            stages[name]["full"]()
            for stage_name in timings:
                timings[stage_name][name].append(
                    time_us(stages[name][stage_name], args.warmup, args.iterations)
                )
    summaries = {}
    for stage_name, stage_profiles in timings.items():
        for name, values in stage_profiles.items():
            value = trimmed(values)
            summaries[(stage_name, name)] = value
            print(
                f"RESULT stage={stage_name} profile={name} median_us="
                f"{statistics.median(values):.3f} trimmed_mean_us={value:.3f}",
                flush=True,
            )
    baseline = summaries[("full", "W8G2080")]
    candidates = {
        name: summaries[("full", name)] for name in profiles[1:]
    }
    best_name = min(candidates, key=candidates.get)
    best = candidates[best_name]
    print(
        f"DECISION baseline_us={baseline:.3f} best={best_name} "
        f"best_us={best:.3f} gain_pct={(baseline / best - 1) * 100:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
