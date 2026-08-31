#!/usr/bin/env python3
"""TP4 DSpark gamma1 M64 expert-row persistent A4 oracle.

Arm A is the accepted M64 row-prefetch stack.  Arm B changes only task
ownership: a gate wave or down subgroup stays on one (expert, row tile) while
it scans that expert's consecutive A4 chunks.  Quantization, FP32 partials and
the fixed-slot reduction are shared and remain separate kernels.  This script
does not alter a production selector.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped_row_prefetch_logical_scale,
    _jit_gate_up_grouped_row_prefetch,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    _per_token_group_quant_int8,
)


E, M, T, H, I, N = 256, 64, 6, 4096, 512, 4096
A, R, GATE_WAVES, DOWN_WAVES = 4, 2, 8, 4
GATE_BLOCKS, DOWN_BLOCKS, LUT = 2080, 832, 2
STAGES = ("gate", "quant", "down", "reduce", "full")


@dataclass(frozen=True)
class ExpertRows:
    active_experts: torch.Tensor
    block_starts: torch.Tensor
    block_counts: torch.Tensor
    num_active: torch.Tensor


@cache_once
def persistent_module():
    args = make_cpp_args(
        E, M, T, I, H, A, R, GATE_WAVES, GATE_BLOCKS, DOWN_BLOCKS
    )
    return load_jit(
        "gfx90a_fp4_expert_row_persistent_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_row_persistent_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_gate",
                f"sglang::Gfx90aFp4ExpertRowPersistentOracle<{args}>::run_gate",
            ),
            (
                "run_down",
                f"sglang::Gfx90aFp4ExpertRowPersistentOracle<{args}>::run_down",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def quant_into(x: torch.Tensor, q: torch.Tensor, scale: torch.Tensor) -> None:
    groups = x.numel() // 32
    _per_token_group_quant_int8[(groups,)](
        x,
        q,
        scale,
        32,
        32,
        1e-10,
        int8_min=-128,
        int8_max=127,
        BLOCK=32,
        num_warps=1,
        num_stages=1,
    )


def build_expert_rows(sorted_experts: torch.Tensor) -> ExpertRows:
    experts = sorted_experts.detach().cpu().to(torch.int32).tolist()
    active: list[int] = []
    starts: list[int] = []
    counts: list[int] = []
    for index, expert in enumerate(experts):
        if not active or expert != active[-1]:
            active.append(expert)
            starts.append(index)
            counts.append(1)
        else:
            counts[-1] += 1
    if any(expert < 0 or expert >= E for expert in active):
        raise RuntimeError("metadata contains an invalid active expert")
    if sum(counts) != len(experts):
        raise RuntimeError("expert block ranges do not cover the A4 metadata")
    device = sorted_experts.device
    return ExpertRows(
        active_experts=torch.tensor(active, dtype=torch.int32, device=device),
        block_starts=torch.tensor(starts, dtype=torch.int32, device=device),
        block_counts=torch.tensor(counts, dtype=torch.int32, device=device),
        num_active=torch.tensor([len(active)], dtype=torch.int32, device=device),
    )


def load_route(path: str, record_index: int, layer: int) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        raise RuntimeError("recorder does not contain a non-empty records list")
    if record_index < 0:
        candidates = [
            (index, record)
            for index, record in enumerate(records)
            if int(record["global_physical_count"][layer].sum()) == M * T
        ]
        if not candidates:
            raise RuntimeError("recorder has no full M64 route")
        record_index, record = candidates[len(candidates) // 2]
    else:
        record = records[record_index]
    counts = record["global_physical_count"][layer].to(torch.int64)
    if counts.shape != (E,) or int(counts.sum()) != M * T:
        raise RuntimeError(
            f"record {record_index} layer {layer} is not M64/top6: "
            f"shape={tuple(counts.shape)} sum={int(counts.sum())}"
        )
    info = {
        "record_index": record_index,
        "forward_pass_id": int(record["forward_pass_id"]),
        "layer": layer,
        "active_experts": int((counts > 0).sum()),
        "assignments": int(counts.sum()),
        "max_occupancy": int(counts.max()),
        "a4_scans": int(((counts + A - 1) // A).sum()),
    }
    return counts, info


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
        default="/tmp/expert_distribution_recorder_1788135603.386394_0.pt",
    )
    parser.add_argument(
        "--record-index",
        type=int,
        default=117,
        help="recorder list index; -1 selects the middle full-M64 record",
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")
    if args.rounds != 7 or args.mutations < 100 or args.graph_replays < 1000:
        raise ValueError("formal run requires 7 rounds, 100 mutations and 1000 replays")

    counts, route = load_route(args.recorder, args.record_index, args.layer)
    topk_ids = reconstruct_topk_from_counts(counts, m=M, topk=T).cuda()
    metadata = make_metadata(topk_ids, assignments=A)
    rows = build_expert_rows(metadata.sorted_experts)
    if int(rows.num_active.item()) != route["active_experts"]:
        raise RuntimeError("active-expert metadata count mismatch")
    if int(rows.block_counts.sum()) != route["a4_scans"]:
        raise RuntimeError("A4 block range count mismatch")
    print("ROUTE " + " ".join(f"{key}={value}" for key, value in route.items()), flush=True)

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq = torch.empty((M, H), dtype=torch.int8, device=device)
    xs = torch.empty((M, H // 32), dtype=torch.float32, device=device)
    quant_into(x, xq, xs)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)

    reference_gate = _jit_gate_up_grouped_row_prefetch(
        E, M, T, I, H, A, R, GATE_WAVES, GATE_BLOCKS, LUT
    )
    reference_down = _jit_down_grouped_row_prefetch_logical_scale(
        E, M, T, N, I, A, DOWN_WAVES, DOWN_BLOCKS, LUT
    )
    candidate = persistent_module()
    states = {}
    stages = {}
    for name in ("A", "B"):
        state = {
            "intermediate": torch.empty((M, T, I), dtype=torch.bfloat16, device=device),
            "iq": torch.empty((M, T, I), dtype=torch.int8, device=device),
            "iscale": torch.empty((M, T, I // 32), dtype=torch.float32, device=device),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device=device),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device=device),
        }

        def gate_stage(name=name, state=state):
            if name == "A":
                reference_gate.run(
                    xq, xs, w13, s13, metadata.sorted_ids,
                    metadata.sorted_experts, metadata.valid,
                    state["intermediate"], 10.0,
                )
            else:
                candidate.run_gate(
                    xq, xs, w13, s13, metadata.sorted_ids,
                    rows.active_experts, rows.block_starts, rows.block_counts,
                    rows.num_active, state["intermediate"], 10.0,
                )

        def quant_stage(state=state):
            quant_into(state["intermediate"], state["iq"], state["iscale"])

        def down_stage(name=name, state=state):
            if name == "A":
                reference_down.run_partial(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, metadata.sorted_experts, metadata.valid,
                    topk_weights, state["partial"],
                )
            else:
                candidate.run_down(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, rows.active_experts,
                    rows.block_starts, rows.block_counts, rows.num_active,
                    topk_weights, state["partial"],
                )

        def reduce_stage(state=state):
            reference_down.reduce(state["partial"], state["output"])

        def full_stage(g=gate_stage, q=quant_stage, d=down_stage, r=reduce_stage):
            g(); q(); d(); r()

        states[name] = state
        stages[name] = {
            "gate": gate_stage, "quant": quant_stage, "down": down_stage,
            "reduce": reduce_stage, "full": full_stage,
        }

    def assert_exact(label: str) -> None:
        for key in ("intermediate", "iq", "iscale", "partial", "output"):
            if not torch.equal(states["A"][key], states["B"][key]):
                delta = (states["A"][key].float() - states["B"][key].float()).abs()
                raise RuntimeError(f"{label} {key} mismatch max_abs={float(delta.max())}")

    for name in ("A", "B"):
        stages[name]["full"]()
    torch.cuda.synchronize(); assert_exact("initial")
    for mutation in range(args.mutations):
        x.normal_(); quant_into(x, xq, xs); topk_weights.uniform_()
        stages["A"]["full"](); stages["B"]["full"]()
        torch.cuda.synchronize(); assert_exact(f"mutation={mutation}")
    print(f"CORRECTNESS eager_mutations={args.mutations} all_exact=True", flush=True)

    graphs = {}
    for name in ("A", "B"):
        for _ in range(3): stages[name]["full"]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): stages[name]["full"]()
        graphs[name] = graph
    for replay in range(args.graph_replays):
        x.normal_(); quant_into(x, xq, xs); topk_weights.uniform_()
        graphs["A"].replay(); graphs["B"].replay()
        torch.cuda.synchronize(); assert_exact(f"graph_replay={replay}")
    print(f"CORRECTNESS graph_replays={args.graph_replays} all_exact=True", flush=True)

    samples = {stage: {"A": [], "B": []} for stage in STAGES}
    for _ in range(args.rounds):
        for name in ("A", "B", "B", "A"):
            stages[name]["full"]()
            for stage in STAGES:
                samples[stage][name].append(
                    time_us(stages[name][stage], args.warmup, args.iterations)
                )
    summary = {}
    for stage in STAGES:
        summary[stage] = {}
        for name in ("A", "B"):
            values = samples[stage][name]
            summary[stage][name] = {
                "median_us": statistics.median(values),
                "trimmed_mean_us": trimmed(values),
                "samples_us": values,
            }
            print(
                f"RESULT stage={stage} arm={name} "
                f"median_us={statistics.median(values):.3f} "
                f"trimmed_us={trimmed(values):.3f}", flush=True,
            )
    a_us = summary["full"]["A"]["trimmed_mean_us"]
    b_us = summary["full"]["B"]["trimmed_mean_us"]
    gain = (a_us / b_us - 1.0) * 100.0
    print(
        f"DECISION baseline_us={a_us:.3f} candidate_us={b_us:.3f} "
        f"gain_pct={gain:.3f} passes_10pct={b_us <= 0.9 * a_us}", flush=True,
    )
    if args.output:
        args.output.write_text(json.dumps({"route": route, "timings": summary}, indent=2) + "\n")
        print(f"REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
