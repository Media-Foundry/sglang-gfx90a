#!/usr/bin/env python3
"""M128 hot-expert LDS-staged A4 routed-FP4 oracle.

The real gamma-three route is split once on the host at occupancy > 8:

* cold experts retain the production A4/R2/W8 G2080/D832 kernels;
* hot gate/up assigns one output row to each wave, decodes that row into
  wave-private LDS once, then walks every A4 chunk for the expert;
* hot down assigns one output row to each subgroup16 and applies the same
  stage-once policy;
* quantization, FP32 slot partials, and the fixed-order BF16 reduction remain
  unchanged.

This is a standalone oracle only and does not install a production selector.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    Metadata,
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    _per_token_group_quant_int8,
)


E, M, T, H, I, N = 256, 128, 6, 4096, 512, 4096
A, R, WAVES, LUT = 4, 2, 8, 2
GATE_BLOCKS, DOWN_BLOCKS = 2080, 832
HOT_WAVES, HOT_GATE_BLOCKS, HOT_DOWN_BLOCKS = 4, 704, 208
HOT_THRESHOLD = 8
ARMS = ("A", "G", "GD")
STAGES = ("gate", "quant", "down", "reduce", "full")


@dataclass(frozen=True)
class ExpertRows:
    active_experts: torch.Tensor
    block_starts: torch.Tensor
    block_counts: torch.Tensor
    num_active: torch.Tensor


@cache_once
def hot_module():
    args = make_cpp_args(
        E,
        M,
        T,
        I,
        H,
        A,
        HOT_WAVES,
        HOT_GATE_BLOCKS,
        HOT_DOWN_BLOCKS,
    )
    return load_jit(
        "gfx90a_fp4_m128_hot_expert_lds_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_hot_expert_lds_oracle.cuh"],
        cuda_wrappers=[
            (
                "run_gate",
                f"sglang::Gfx90aFp4HotExpertLdsOracle<{args}>::run_gate",
            ),
            (
                "run_down",
                f"sglang::Gfx90aFp4HotExpertLdsOracle<{args}>::run_down",
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


def load_real_counts(path: Path, layer: int) -> tuple[torch.Tensor, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("expected a per-pass recorder with a records list")
    matches = [
        (index, record)
        for index, record in enumerate(records)
        if int(record["global_physical_count"][layer].sum()) == M * T
    ]
    if not matches:
        raise RuntimeError(f"no M{M}/top-{T} target record in {path}")
    index, record = matches[len(matches) // 2]
    counts = record["global_physical_count"][layer].to(torch.int64)
    hot = counts > HOT_THRESHOLD
    info = {
        "record_index": index,
        "forward_pass_id": int(record["forward_pass_id"]),
        "layer": layer,
        "active_experts": int((counts > 0).sum()),
        "max_occupancy": int(counts.max()),
        "a4_scans": int(((counts + A - 1) // A).sum()),
        "hot_threshold": HOT_THRESHOLD,
        "hot_experts": int(hot.sum()),
        "hot_assignments": int(counts[hot].sum()),
        "cold_assignments": int(counts[~hot].sum()),
        "hot_a4_scans": int(((counts[hot] + A - 1) // A).sum()),
        "cold_a4_scans": int(
            ((counts[(counts > 0) & ~hot] + A - 1) // A).sum()
        ),
    }
    return counts, info


def make_subset_metadata(
    topk_ids: torch.Tensor, hot_mask: torch.Tensor, *, select_hot: bool
) -> Metadata:
    buckets: list[list[int]] = [[] for _ in range(E)]
    hot_list = hot_mask.to(torch.bool).cpu().tolist()
    for token, experts in enumerate(topk_ids.cpu().tolist()):
        for slot, expert in enumerate(experts):
            if bool(hot_list[expert]) == select_hot:
                buckets[expert].append((slot << 24) | token)
    sentinel = M
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        if not bucket:
            continue
        for offset in range(0, len(bucket), A):
            block = bucket[offset : offset + A]
            ids.extend(block)
            ids.extend([sentinel] * (A - len(block)))
            experts.append(expert)
    device = topk_ids.device
    return Metadata(
        assignments=A,
        sorted_ids=torch.tensor(ids, dtype=torch.int32, device=device),
        sorted_experts=torch.tensor(experts, dtype=torch.int32, device=device),
        valid=torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
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
    if not active or sum(counts) != len(experts):
        raise RuntimeError("invalid hot-expert A4 ranges")
    device = sorted_experts.device
    return ExpertRows(
        active_experts=torch.tensor(active, dtype=torch.int32, device=device),
        block_starts=torch.tensor(starts, dtype=torch.int32, device=device),
        block_counts=torch.tensor(counts, dtype=torch.int32, device=device),
        num_active=torch.tensor([len(active)], dtype=torch.int32, device=device),
    )


def trimmed(values: list[float]) -> float:
    return statistics.fmean(sorted(values)[1:-1])


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
    parser.add_argument(
        "--recorder",
        type=Path,
        default=Path(
            "/tmp/dsv4_gamma3_recorder2/"
            "expert_distribution_recorder_1788146391.4818711_0.pt"
        ),
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.rounds != 7 or args.mutations < 100 or args.graph_replays < 1000:
        raise ValueError("formal oracle requires 7 rounds, 100 mutations, 1000 replays")
    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")

    counts, route = load_real_counts(args.recorder, args.layer)
    topk_ids = reconstruct_topk_from_counts(counts, m=M, topk=T).cuda()
    full_metadata = make_metadata(topk_ids, assignments=A)
    hot_mask = counts > HOT_THRESHOLD
    cold_metadata = make_subset_metadata(topk_ids, hot_mask, select_hot=False)
    hot_metadata = make_subset_metadata(topk_ids, hot_mask, select_hot=True)
    hot_rows = build_expert_rows(hot_metadata.sorted_experts)
    if (
        cold_metadata.sorted_experts.numel() + hot_metadata.sorted_experts.numel()
        != full_metadata.sorted_experts.numel()
    ):
        raise RuntimeError("cold/hot A4 scans do not partition the full metadata")
    if int(hot_rows.num_active.item()) != route["hot_experts"]:
        raise RuntimeError("hot-expert count mismatch")
    if int(hot_rows.block_counts.sum()) != route["hot_a4_scans"]:
        raise RuntimeError("hot A4 range count mismatch")
    print("ROUTE " + " ".join(f"{k}={v}" for k, v in route.items()), flush=True)

    torch.manual_seed(20260831)
    device = torch.device("cuda")
    x = torch.randn((M, H), dtype=torch.bfloat16, device=device)
    xq = torch.empty((M, H), dtype=torch.int8, device=device)
    xs = torch.empty((M, H // 32), dtype=torch.float32, device=device)
    quant_into(x, xq, xs)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    w13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device
    )
    s13 = torch.full(
        (E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device
    )
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)

    reference_gate = _jit_gate_up_grouped(
        E, M, T, I, H, A, R, WAVES, GATE_BLOCKS, LUT
    )
    reference_down = _jit_down_grouped(
        E, M, T, N, I, A, R, WAVES, DOWN_BLOCKS, LUT
    )
    candidate = hot_module()
    states: dict[str, dict[str, torch.Tensor]] = {}
    stages: dict[str, dict[str, object]] = {}
    for name in ARMS:
        state = {
            "intermediate": torch.empty(
                (M, T, I), dtype=torch.bfloat16, device=device
            ),
            "iq": torch.empty((M, T, I), dtype=torch.int8, device=device),
            "iscale": torch.empty(
                (M, T, I // 32), dtype=torch.float32, device=device
            ),
            "partial": torch.empty((M, T, N), dtype=torch.float32, device=device),
            "output": torch.empty((M, N), dtype=torch.bfloat16, device=device),
        }

        def gate_stage(name=name, state=state):
            metadata = full_metadata if name == "A" else cold_metadata
            reference_gate.run(
                xq,
                xs,
                w13,
                s13,
                metadata.sorted_ids,
                metadata.sorted_experts,
                metadata.valid,
                state["intermediate"],
                10.0,
            )
            if name != "A":
                candidate.run_gate(
                    xq,
                    xs,
                    w13,
                    s13,
                    hot_metadata.sorted_ids,
                    hot_rows.active_experts,
                    hot_rows.block_starts,
                    hot_rows.block_counts,
                    hot_rows.num_active,
                    state["intermediate"],
                    10.0,
                )

        def quant_stage(state=state):
            quant_into(state["intermediate"], state["iq"], state["iscale"])

        def down_stage(name=name, state=state):
            metadata = cold_metadata if name == "GD" else full_metadata
            reference_down.run_partial(
                state["iq"],
                state["iscale"],
                w2,
                s2,
                metadata.sorted_ids,
                metadata.sorted_experts,
                metadata.valid,
                topk_weights,
                state["partial"],
            )
            if name == "GD":
                candidate.run_down(
                    state["iq"],
                    state["iscale"],
                    w2,
                    s2,
                    hot_metadata.sorted_ids,
                    hot_rows.active_experts,
                    hot_rows.block_starts,
                    hot_rows.block_counts,
                    hot_rows.num_active,
                    topk_weights,
                    state["partial"],
                )

        def reduce_stage(state=state):
            reference_down.reduce(state["partial"], state["output"])

        def full_stage(g=gate_stage, q=quant_stage, d=down_stage, r=reduce_stage):
            g()
            q()
            d()
            r()

        states[name] = state
        stages[name] = {
            "gate": gate_stage,
            "quant": quant_stage,
            "down": down_stage,
            "reduce": reduce_stage,
            "full": full_stage,
        }

    def assert_exact(label: str) -> None:
        for tensor_name in ("intermediate", "iq", "iscale", "partial", "output"):
            expected = states["A"][tensor_name]
            for name in ("G", "GD"):
                actual = states[name][tensor_name]
                if not torch.equal(expected, actual):
                    delta = (expected.float() - actual.float()).abs()
                    mismatch = torch.nonzero(delta, as_tuple=False)
                    raise RuntimeError(
                        f"{label} arm={name} tensor={tensor_name} "
                        f"max_abs={float(delta.max())} mismatches={mismatch.shape[0]} "
                        f"first={mismatch[:8].cpu().tolist()}"
                    )

    for name in ARMS:
        stages[name]["full"]()
    torch.cuda.synchronize()
    assert_exact("initial")
    for mutation in range(args.mutations):
        x.normal_()
        quant_into(x, xq, xs)
        topk_weights.uniform_()
        for name in ARMS:
            stages[name]["full"]()
        torch.cuda.synchronize()
        assert_exact(f"mutation={mutation}")
    print(f"CORRECT eager_mutations={args.mutations} bitwise_exact=True", flush=True)

    graphs: dict[str, torch.cuda.CUDAGraph] = {}
    for name in ARMS:
        for _ in range(3):
            stages[name]["full"]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stages[name]["full"]()
        graphs[name] = graph
    for replay in range(args.graph_replays):
        x.normal_()
        quant_into(x, xq, xs)
        topk_weights.uniform_()
        for name in ARMS:
            graphs[name].replay()
        torch.cuda.synchronize()
        assert_exact(f"graph_replay={replay}")
    print(f"CORRECT graph_replays={args.graph_replays} bitwise_exact=True", flush=True)

    samples = {
        stage: {name: [] for name in ARMS}
        for stage in STAGES
    }
    for _ in range(args.rounds):
        for name in ("A", "G", "GD", "GD", "G", "A"):
            stages[name]["full"]()
            for stage in STAGES:
                samples[stage][name].append(
                    time_us(stages[name][stage], args.warmup, args.iterations)
                )
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for stage in STAGES:
        summary[stage] = {}
        for name in ARMS:
            values = samples[stage][name]
            summary[stage][name] = {
                "median_us": statistics.median(values),
                "trimmed_mean_us": trimmed(values),
                "samples_us": values,
            }
            print(
                f"RESULT stage={stage} arm={name} "
                f"median_us={statistics.median(values):.3f} "
                f"trimmed_us={trimmed(values):.3f}",
                flush=True,
            )
    base_gate = float(summary["gate"]["A"]["trimmed_mean_us"])
    mixed_gate = float(summary["gate"]["G"]["trimmed_mean_us"])
    base_full = float(summary["full"]["A"]["trimmed_mean_us"])
    gate_full = float(summary["full"]["G"]["trimmed_mean_us"])
    both_full = float(summary["full"]["GD"]["trimmed_mean_us"])
    decisions = {
        "gate_gain_pct": (base_gate / mixed_gate - 1.0) * 100.0,
        "gate_only_full_gain_pct": (base_full / gate_full - 1.0) * 100.0,
        "gate_down_full_gain_pct": (base_full / both_full - 1.0) * 100.0,
    }
    decisions["passes_15pct"] = max(
        decisions["gate_gain_pct"], decisions["gate_down_full_gain_pct"]
    ) >= 15.0
    print(
        "DECISION " + " ".join(f"{k}={v}" for k, v in decisions.items()),
        flush=True,
    )
    if args.output:
        payload = {"route": route, "timings": summary, "decision": decisions}
        args.output.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
