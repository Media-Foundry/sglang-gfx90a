#!/usr/bin/env python3
"""Oracle-only TP4 DSpark M64 CTA weight-multicast benchmark.

The candidate partitions the real stable A4 metadata into cold singleton
chunks and hot descriptors containing 2--4 consecutive chunks of one expert.
Cold chunks use the accepted kernels; hot chunks use the CTA multicast core.
No production selector imports this file.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_dspark_gamma1_m64_expert_row_persistent import (
    A,
    DOWN_BLOCKS,
    DOWN_WAVES,
    E,
    GATE_BLOCKS,
    GATE_WAVES,
    H,
    I,
    LUT,
    M,
    N,
    R,
    T,
    load_route,
    quant_into,
    time_us,
    trimmed,
)
from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    Metadata,
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_cta_weight_multicast_oracle import (
    run_down as multicast_down,
)
from sglang.kernels.ops.moe.gfx90a_fp4_cta_weight_multicast_oracle import (
    run_gate as multicast_gate,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped_row_prefetch_logical_scale,
    _jit_gate_up_grouped_row_prefetch,
)


STAGES = ("gate", "quant", "down", "reduce", "full")


@dataclass(frozen=True)
class Partition:
    cold: Metadata
    descriptor_experts: torch.Tensor
    descriptor_starts: torch.Tensor
    descriptor_counts: torch.Tensor
    num_descriptors: torch.Tensor
    hot_blocks: int
    cold_blocks: int
    physical_loads: int


def build_partition(metadata: Metadata) -> Partition:
    """Preserve every encoded partial slot exactly once."""
    experts = metadata.sorted_experts.detach().cpu().to(torch.int32).tolist()
    descriptor_experts: list[int] = []
    descriptor_starts: list[int] = []
    descriptor_counts: list[int] = []
    hot_indices: list[int] = []
    cold_indices: list[int] = []

    begin = 0
    while begin < len(experts):
        expert = experts[begin]
        end = begin + 1
        while end < len(experts) and experts[end] == expert:
            end += 1
        cursor = begin
        remaining = end - begin
        while remaining >= 2:
            count = min(4, remaining)
            descriptor_experts.append(expert)
            descriptor_starts.append(cursor)
            descriptor_counts.append(count)
            hot_indices.extend(range(cursor, cursor + count))
            cursor += count
            remaining -= count
        if remaining == 1:
            cold_indices.append(cursor)
        begin = end

    all_indices = sorted(hot_indices + cold_indices)
    if all_indices != list(range(len(experts))):
        raise RuntimeError("hot/cold descriptors do not partition A4 blocks")
    if len(hot_indices) != sum(descriptor_counts):
        raise RuntimeError("descriptor chunk counts do not cover hot blocks")

    ids_cpu = metadata.sorted_ids.detach().cpu().view(-1, A)
    valid_slots = []
    for block in all_indices:
        for encoded in ids_cpu[block].tolist():
            token = encoded & 0x00FFFFFF
            slot = (encoded & 0xFFFFFFFF) >> 24
            if token < M and slot < T:
                valid_slots.append((token, slot))
    if len(valid_slots) != M * T or len(set(valid_slots)) != M * T:
        raise RuntimeError("descriptor partition duplicates or drops a partial slot")

    device = metadata.sorted_ids.device
    if cold_indices:
        index = torch.tensor(cold_indices, dtype=torch.int64, device=device)
        cold_ids = metadata.sorted_ids.view(-1, A).index_select(0, index).reshape(-1)
        cold_experts = metadata.sorted_experts.index_select(0, index)
    else:
        cold_ids = torch.empty(0, dtype=torch.int32, device=device)
        cold_experts = torch.empty(0, dtype=torch.int32, device=device)
    cold = Metadata(
        A,
        cold_ids,
        cold_experts,
        torch.tensor([cold_ids.numel(), 0], dtype=torch.int32, device=device),
    )
    return Partition(
        cold=cold,
        descriptor_experts=torch.tensor(
            descriptor_experts, dtype=torch.int32, device=device
        ),
        descriptor_starts=torch.tensor(
            descriptor_starts, dtype=torch.int32, device=device
        ),
        descriptor_counts=torch.tensor(
            descriptor_counts, dtype=torch.int32, device=device
        ),
        num_descriptors=torch.tensor(
            [len(descriptor_counts)], dtype=torch.int32, device=device
        ),
        hot_blocks=len(hot_indices),
        cold_blocks=len(cold_indices),
        physical_loads=len(descriptor_counts) + len(cold_indices),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--recorder",
        default="/tmp/expert_distribution_recorder_1788135603.386394_0.pt",
    )
    parser.add_argument("--record-index", type=int, default=117)
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
    partition = build_partition(metadata)
    reduction = 1.0 - partition.physical_loads / route["a4_scans"]
    route.update(
        hot_blocks=partition.hot_blocks,
        cold_blocks=partition.cold_blocks,
        multicast_descriptors=int(partition.num_descriptors.item()),
        multicast_physical_loads=partition.physical_loads,
        load_reduction=reduction,
    )
    if reduction < 0.20:
        raise RuntimeError(f"route load reduction {reduction:.3%} misses 20% gate")
    print("ROUTE " + " ".join(f"{k}={v}" for k, v in route.items()), flush=True)

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

    gate = _jit_gate_up_grouped_row_prefetch(
        E, M, T, I, H, A, R, GATE_WAVES, GATE_BLOCKS, LUT
    )
    down = _jit_down_grouped_row_prefetch_logical_scale(
        E, M, T, N, I, A, DOWN_WAVES, DOWN_BLOCKS, LUT
    )
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
            md = metadata if name == "A" else partition.cold
            if md.sorted_experts.numel():
                gate.run(
                    xq, xs, w13, s13, md.sorted_ids, md.sorted_experts,
                    md.valid, state["intermediate"], 10.0,
                )
            if name == "B":
                multicast_gate(
                    xq, xs, w13, s13, metadata.sorted_ids,
                    partition.descriptor_experts, partition.descriptor_starts,
                    partition.descriptor_counts, partition.num_descriptors,
                    state["intermediate"], 10.0,
                )

        def quant_stage(state=state):
            quant_into(state["intermediate"], state["iq"], state["iscale"])

        def down_stage(name=name, state=state):
            md = metadata if name == "A" else partition.cold
            if md.sorted_experts.numel():
                down.run_partial(
                    state["iq"], state["iscale"], w2, s2, md.sorted_ids,
                    md.sorted_experts, md.valid, topk_weights, state["partial"],
                )
            if name == "B":
                multicast_down(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, partition.descriptor_experts,
                    partition.descriptor_starts, partition.descriptor_counts,
                    partition.num_descriptors, topk_weights, state["partial"],
                )

        def reduce_stage(state=state):
            down.reduce(state["partial"], state["output"])

        def full_stage(g=gate_stage, q=quant_stage, d=down_stage, r=reduce_stage):
            g(); q(); d(); r()

        states[name] = state
        stages[name] = dict(
            gate=gate_stage, quant=quant_stage, down=down_stage,
            reduce=reduce_stage, full=full_stage,
        )

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

    samples = {stage: {"A": [], "B": []} for stage in STAGES}
    for _ in range(args.rounds):
        for name in ("A", "B", "B", "A"):
            stages[name]["full"]()
            for stage in STAGES:
                samples[stage][name].append(
                    time_us(stages[name][stage], args.warmup, args.iterations)
                )
    summary = {
        stage: {
            name: {
                "median_us": statistics.median(samples[stage][name]),
                "trimmed_mean_us": trimmed(samples[stage][name]),
                "samples_us": samples[stage][name],
            }
            for name in ("A", "B")
        }
        for stage in STAGES
    }
    a_us = summary["full"]["A"]["trimmed_mean_us"]
    b_us = summary["full"]["B"]["trimmed_mean_us"]
    saving = a_us - b_us
    report = {
        "route": route,
        "timings": summary,
        "saving_us": saving,
        "passes_50us": saving >= 50.0,
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    if saving < 50.0:
        raise RuntimeError(f"multicast saving {saving:.3f} us misses 50-us gate")


if __name__ == "__main__":
    main()
