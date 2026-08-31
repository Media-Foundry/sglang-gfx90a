#!/usr/bin/env python3
"""Exact A4/A8 occupancy-hybrid oracle for DSpark gamma-1 target M64."""

from __future__ import annotations

import argparse
import glob
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    reconstruct_topk_from_counts,
)
from scripts.rocm.bench_dsv4_tp4_dspark_m84_specialization import (
    quant_into,
    time_us,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped_row_prefetch_logical_scale,
    _jit_gate_up_grouped_row_prefetch,
)

E, M, T, H, I, N = 256, 64, 6, 4096, 512, 4096
ROWS, WAVES, LDS = 2, 8, 2


@dataclass(frozen=True)
class Metadata:
    ids: torch.Tensor
    experts: torch.Tensor
    valid: torch.Tensor


@dataclass
class State:
    intermediate: torch.Tensor
    iq: torch.Tensor
    iscale: torch.Tensor
    partial: torch.Tensor
    output: torch.Tensor


def make_metadata(
    topk: torch.Tensor, *, assignments: int, high_occupancy: bool | None
) -> Metadata:
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, row in enumerate(topk.cpu().tolist()):
        for slot, expert in enumerate(row):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        if not bucket:
            continue
        if high_occupancy is not None and ((len(bucket) > 4) != high_occupancy):
            continue
        for offset in range(0, len(bucket), assignments):
            block = bucket[offset : offset + assignments]
            ids.extend(block + [M] * (assignments - len(block)))
            experts.append(expert)
    device = topk.device
    return Metadata(
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def allocate_state(device: torch.device) -> State:
    return State(
        torch.empty((M, T, I), dtype=torch.bfloat16, device=device),
        torch.empty((M, T, I), dtype=torch.int8, device=device),
        torch.empty((M, T, I // 32), dtype=torch.float32, device=device),
        torch.empty((M, T, N), dtype=torch.float32, device=device),
        torch.empty((M, N), dtype=torch.bfloat16, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--full-pass-index", type=int, default=-1)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds < 7:
        raise ValueError("formal oracle requires 100 mutations, 1000 replays, 7 rounds")
    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    full = [
        record
        for record in payload["records"]
        if int(record["global_physical_count"][args.layer].sum()) == M * T
    ]
    if not full:
        raise RuntimeError("recorder contains no full M64 target pass")
    record = full[args.full_pass_index]
    counts = record["global_physical_count"][args.layer].to(torch.int64)
    topk = reconstruct_topk_from_counts(counts, m=M, topk=T).cuda()
    all_a4 = make_metadata(topk, assignments=4, high_occupancy=None)
    low_a4 = make_metadata(topk, assignments=4, high_occupancy=False)
    high_a8 = make_metadata(topk, assignments=8, high_occupancy=True)
    print(
        "ROUTE "
        f"pass={record['forward_pass_id']} active={int((counts > 0).sum())} "
        f"a4_scans={all_a4.experts.numel()} low_a4={low_a4.experts.numel()} "
        f"high_a8={high_a8.experts.numel()} "
        f"hybrid_scans={low_a4.experts.numel() + high_a8.experts.numel()} "
        f"high_assignments={int(counts[counts > 4].sum())}",
        flush=True,
    )

    torch.manual_seed(831)
    device = torch.device("cuda")
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq = torch.empty((M, H), dtype=torch.int8, device=device)
    xscale = torch.empty((M, H // 32), dtype=torch.float32, device=device)
    quant_into(x, xq, xscale)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)

    def build_runner(
        *, low_gate: int, low_down: int, high_gate: int, high_down: int
    ):
        state = allocate_state(device)
        gate_low = _jit_gate_up_grouped_row_prefetch(
            E, M, T, I, H, 4, ROWS, WAVES, low_gate, LDS
        )
        gate_high = _jit_gate_up_grouped_row_prefetch(
            E, M, T, I, H, 8, ROWS, WAVES, high_gate, LDS
        )
        down_low = _jit_down_grouped_row_prefetch_logical_scale(
            E, M, T, N, I, 4, 4, low_down, LDS
        )
        down_high = _jit_down_grouped_row_prefetch_logical_scale(
            E, M, T, N, I, 8, 4, high_down, LDS
        )

        def run() -> None:
            gate_low.run(
                xq, xscale, w13, s13, low_a4.ids, low_a4.experts,
                low_a4.valid, state.intermediate, 10.0
            )
            gate_high.run(
                xq, xscale, w13, s13, high_a8.ids, high_a8.experts,
                high_a8.valid, state.intermediate, 10.0
            )
            quant_into(state.intermediate, state.iq, state.iscale)
            down_low.run_partial(
                state.iq, state.iscale, w2, s2, low_a4.ids, low_a4.experts,
                low_a4.valid, topk_weights, state.partial
            )
            down_high.run_partial(
                state.iq, state.iscale, w2, s2, high_a8.ids, high_a8.experts,
                high_a8.valid, topk_weights, state.partial
            )
            down_low.reduce(state.partial, state.output)

        return run, state

    baseline = allocate_state(device)
    gate_a = _jit_gate_up_grouped_row_prefetch(E, M, T, I, H, 4, 2, 8, 2080, LDS)
    down_a = _jit_down_grouped_row_prefetch_logical_scale(
        E, M, T, N, I, 4, 4, 832, LDS
    )

    def run_a() -> None:
        gate_a.run(
            xq, xscale, w13, s13, all_a4.ids, all_a4.experts,
            all_a4.valid, baseline.intermediate, 10.0
        )
        quant_into(baseline.intermediate, baseline.iq, baseline.iscale)
        down_a.run_partial(
            baseline.iq, baseline.iscale, w2, s2, all_a4.ids,
            all_a4.experts, all_a4.valid, topk_weights, baseline.partial
        )
        down_a.reduce(baseline.partial, baseline.output)

    configs = {
        "L832_416_H1664_832": build_runner(
            low_gate=832, low_down=416, high_gate=1664, high_down=832
        ),
        "L832_416_H2080_832": build_runner(
            low_gate=832, low_down=416, high_gate=2080, high_down=832
        ),
        "L1040_624_H1664_1248": build_runner(
            low_gate=1040, low_down=624, high_gate=1664, high_down=1248
        ),
        "L1040_624_H2080_1248": build_runner(
            low_gate=1040, low_down=624, high_gate=2080, high_down=1248
        ),
    }
    all_runners = {"A4": (run_a, baseline), **configs}

    for mutation in range(args.mutations):
        x.normal_()
        topk_weights.uniform_()
        quant_into(x, xq, xscale)
        for run, _state in all_runners.values():
            run()
        torch.cuda.synchronize()
        for name, (_run, state) in configs.items():
            for field in ("intermediate", "iq", "iscale", "partial", "output"):
                actual = getattr(state, field)
                expected = getattr(baseline, field)
                if not torch.equal(actual, expected):
                    error = float((actual.float() - expected.float()).abs().max())
                    raise RuntimeError(
                        f"mutation={mutation} {name}.{field} max_abs={error}"
                    )
    print(f"CORRECT mutations={args.mutations} exact=true", flush=True)

    graphs = {}
    for name, (run, _state) in all_runners.items():
        for _ in range(5):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        graphs[name] = graph
    for _ in range(args.graph_replays):
        for graph in graphs.values():
            graph.replay()
    torch.cuda.synchronize()
    for name, (_run, state) in configs.items():
        if not torch.equal(state.output, baseline.output):
            raise RuntimeError(f"graph replay mismatch for {name}")
    print(f"GRAPH replays={args.graph_replays} exact=true", flush=True)

    timings = {name: [] for name in all_runners}
    names = list(all_runners)
    for _ in range(args.rounds):
        for name in names + list(reversed(names)):
            timings[name].append(
                time_us(all_runners[name][0], warmup=20, iterations=30)
            )
    for name, values in timings.items():
        ordered = sorted(values)
        print(
            f"RESULT {name} median_us={statistics.median(values):.3f} "
            f"trimmed_us={statistics.fmean(ordered[1:-1]):.3f} samples="
            + ",".join(f"{value:.3f}" for value in values),
            flush=True,
        )


if __name__ == "__main__":
    main()
