#!/usr/bin/env python3
"""Standalone strict-target TP4 A4 same-expert wave-pod oracle.

Arm A is the existing barrier-free direct-decode grouped kernel.  Arm B keeps
the same per-wave A4 arithmetic and fixed FP32 partial/reduction order, but
places up to four consecutive A4 blocks of one expert in the four waves of a
CTA.  B uses no LDS payload, barrier, atomic queue, or host synchronization.

The benchmark accepts a real expert-distribution recorder or produces an
injectable synthetic route.  Metadata lives in fixed-address device buffers,
so route mutations also exercise HIP Graph replay rather than rebuilding the
graph.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
from scripts.rocm.bench_dsv4_dspark_m128_geometry import quant_into
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_wave_pod_oracle import (
    jit_wave_pod_oracle,
)


E, T, H, I, N = 256, 6, 4096, 512, 4096
A, R, BASE_GATE_WAVES, BASE_DOWN_WAVES = 4, 2, 8, 4
GATE_BLOCKS, DOWN_BLOCKS, DIRECT = 2080, 832, 0
POD_WAVES = 4
STAGES = ("gate", "quant", "down", "reduce", "full")


@dataclass
class StaticMetadata:
    sorted_ids: torch.Tensor
    sorted_experts: torch.Tensor
    valid: torch.Tensor
    pod_blocks: torch.Tensor
    num_pods: torch.Tensor


def trimmed(values: list[float]) -> float:
    if len(values) < 3:
        return statistics.fmean(values)
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


def load_real_route(path: Path, m: int, layer: int, record_index: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    matches: list[torch.Tensor] = []
    for record in payload.get("records", []):
        topk = record.get("topk_ids_of_layer")
        if not isinstance(topk, torch.Tensor) or topk.ndim != 3:
            continue
        if topk.shape[1:] != (m, T):
            continue
        candidate = topk[layer].to(torch.int32).contiguous()
        if bool(((candidate >= 0) & (candidate < E)).all()):
            matches.append(candidate)
    if not matches:
        raise RuntimeError(f"no full M{m} route for layer {layer} in {path}")
    index = record_index if record_index >= 0 else len(matches) // 2
    if index >= len(matches):
        raise ValueError(f"record-index {index} outside {len(matches)} matching routes")
    return matches[index]


def synthetic_route(m: int, active: int, seed: int) -> torch.Tensor:
    """Deterministic diverse route with no duplicate expert inside a token."""
    if active < T or active > E:
        raise ValueError(f"active experts must be in [{T},{E}]")
    generator = torch.Generator().manual_seed(seed)
    hot = torch.randperm(E, generator=generator)[:active]
    rows = []
    # A Zipf-like rotating head gives both singleton and repeated A4 chunks.
    ranks = torch.arange(active, dtype=torch.float64)
    probs = 1.0 / torch.pow(ranks + 2.0, 0.70)
    probs /= probs.sum()
    for token in range(m):
        choice = torch.multinomial(probs, T, replacement=False, generator=generator)
        rows.append(hot[choice])
    return torch.stack(rows).to(torch.int32)


def allocate_metadata(m: int, device: torch.device) -> StaticMetadata:
    # Sorting pads each expert independently to A4.  In the extreme every
    # useful assignment can occupy its own block, so M*T/A is not a safe
    # capacity even though there are only M*T useful assignments.
    max_blocks = m * T
    max_pods = E * ((m + 15) // 16)
    return StaticMetadata(
        sorted_ids=torch.full((max_blocks * A,), m, dtype=torch.int32, device=device),
        sorted_experts=torch.full((max_blocks,), -1, dtype=torch.int32, device=device),
        valid=torch.zeros((2,), dtype=torch.int32, device=device),
        pod_blocks=torch.full((max_pods, POD_WAVES), -1, dtype=torch.int32, device=device),
        num_pods=torch.zeros((1,), dtype=torch.int32, device=device),
    )


def build_pods(sorted_experts: torch.Tensor, max_pods: int) -> torch.Tensor:
    experts = sorted_experts.detach().cpu().to(torch.int32).tolist()
    pods: list[list[int]] = []
    start = 0
    while start < len(experts):
        expert = experts[start]
        end = start + 1
        while end < len(experts) and experts[end] == expert:
            end += 1
        for chunk in range(start, end, POD_WAVES):
            block_ids = list(range(chunk, min(chunk + POD_WAVES, end)))
            pods.append(block_ids + [-1] * (POD_WAVES - len(block_ids)))
        start = end
    if len(pods) > max_pods:
        raise RuntimeError(f"pod count {len(pods)} exceeds static capacity {max_pods}")
    out = torch.full((max_pods, POD_WAVES), -1, dtype=torch.int32)
    if pods:
        out[: len(pods)] = torch.tensor(pods, dtype=torch.int32)
    return out


def inject_route(buffers: StaticMetadata, topk: torch.Tensor) -> dict[str, float | int]:
    metadata = make_metadata(topk, assignments=A)
    blocks = metadata.sorted_experts.numel()
    buffers.sorted_ids.fill_(topk.shape[0])
    buffers.sorted_experts.fill_(-1)
    buffers.sorted_ids[: metadata.sorted_ids.numel()].copy_(metadata.sorted_ids)
    buffers.sorted_experts[:blocks].copy_(metadata.sorted_experts)
    buffers.valid.copy_(metadata.valid)
    pods = build_pods(metadata.sorted_experts, buffers.pod_blocks.shape[0])
    buffers.pod_blocks.copy_(pods.to(buffers.pod_blocks.device))
    num_pods = int(torch.count_nonzero(pods[:, 0] >= 0))
    buffers.num_pods.fill_(num_pods)
    active = int(torch.unique(topk).numel())
    repeat = blocks - active
    return {
        "active_experts": active,
        "a4_blocks": blocks,
        "repeat_chunks": repeat,
        "repeat_fraction": repeat / blocks if blocks else 0.0,
        "pods": num_pods,
    }


def assert_same(states: dict, label: str) -> None:
    for key in ("intermediate", "iq", "iscale", "partial", "output"):
        a = states["A"][key]
        b = states["B"][key]
        if not torch.equal(a, b):
            delta = (a.float() - b.float()).abs()
            raise RuntimeError(
                f"{label}: {key} mismatch max_abs={float(delta.max())} "
                f"count={int(torch.count_nonzero(delta))}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(64, 96, 128), default=96)
    parser.add_argument("--recorder", type=Path)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--record-index", type=int, default=-1)
    parser.add_argument("--synthetic-active", type=int, default=90)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 7:
        raise ValueError("formal run requires >=100 mutations, >=1000 replays, 7 rounds")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("wave-pod oracle requires gfx90a")

    device = torch.device("cuda")
    m = args.m
    topk_cpu = (
        load_real_route(args.recorder, m, args.layer, args.record_index)
        if args.recorder
        else synthetic_route(m, args.synthetic_active, 20260901)
    )
    topk = topk_cpu.to(device)
    metadata = allocate_metadata(m, device)
    route_summary = inject_route(metadata, topk)
    print(f"ROUTE M={m} layer={args.layer} {route_summary}", flush=True)

    torch.manual_seed(20260901)
    x = torch.randn((m, H), dtype=torch.bfloat16, device=device)
    topk_weights = torch.rand((m, T), dtype=torch.float32, device=device)
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device)
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device=device)
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device=device)

    baseline_gate = _jit_gate_up_grouped(
        E, m, T, I, H, A, R, BASE_GATE_WAVES, GATE_BLOCKS, DIRECT
    )
    baseline_down = _jit_down_grouped(
        E, m, T, N, I, A, R, BASE_DOWN_WAVES, DOWN_BLOCKS, DIRECT
    )
    candidate = jit_wave_pod_oracle(
        E, m, T, I, H, A, R, GATE_BLOCKS, DOWN_BLOCKS
    )

    states: dict[str, dict[str, torch.Tensor]] = {}
    stages = {}
    for name in ("A", "B"):
        state = {
            "xq": torch.empty((m, H), dtype=torch.int8, device=device),
            "xscale": torch.empty((m, H // 32), dtype=torch.float32, device=device),
            "intermediate": torch.zeros((m, T, I), dtype=torch.bfloat16, device=device),
            "iq": torch.empty((m, T, I), dtype=torch.int8, device=device),
            "iscale": torch.empty((m, T, I // 32), dtype=torch.float32, device=device),
            "partial": torch.zeros((m, T, N), dtype=torch.float32, device=device),
            "output": torch.empty((m, N), dtype=torch.bfloat16, device=device),
        }

        def gate_stage(name=name, state=state):
            quant_into(x, state["xq"], state["xscale"])
            if name == "A":
                baseline_gate.run(
                    state["xq"], state["xscale"], w13, s13,
                    metadata.sorted_ids, metadata.sorted_experts, metadata.valid,
                    state["intermediate"], 10.0,
                )
            else:
                candidate.run_gate(
                    state["xq"], state["xscale"], w13, s13,
                    metadata.sorted_ids, metadata.sorted_experts,
                    metadata.pod_blocks, metadata.num_pods,
                    state["intermediate"], 10.0,
                )

        def quant_stage(state=state):
            quant_into(state["intermediate"], state["iq"], state["iscale"])

        def down_stage(name=name, state=state):
            state["partial"].zero_()
            if name == "A":
                baseline_down.run_partial(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, metadata.sorted_experts, metadata.valid,
                    topk_weights, state["partial"],
                )
            else:
                candidate.run_down(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, metadata.sorted_experts,
                    metadata.pod_blocks, metadata.num_pods,
                    topk_weights, state["partial"],
                )

        def reduce_stage(state=state):
            baseline_down.reduce(state["partial"], state["output"])

        def full_stage(g=gate_stage, q=quant_stage, d=down_stage, r=reduce_stage):
            g(); q(); d(); r()

        states[name] = state
        stages[name] = {
            "gate": gate_stage,
            "quant": quant_stage,
            "down": down_stage,
            "reduce": reduce_stage,
            "full": full_stage,
        }

    for name in ("A", "B"):
        stages[name]["full"]()
    torch.cuda.synchronize()
    assert_same(states, "initial")

    route_summaries = [route_summary]
    for mutation in range(args.mutations):
        x.normal_()
        topk_weights.uniform_()
        # Exercise live route metadata as well as activations.  Static buffers
        # preserve the pointers later captured by both graphs.
        varied = synthetic_route(
            m,
            max(T, min(E, args.synthetic_active + mutation % 17 - 8)),
            20260902 + mutation,
        ).to(device)
        route_summaries.append(inject_route(metadata, varied))
        stages["A"]["full"](); stages["B"]["full"]()
        torch.cuda.synchronize()
        assert_same(states, f"eager mutation={mutation}")
    # Restore the measured route for graph timing.
    inject_route(metadata, topk)
    print(f"CORRECT eager_mutations={args.mutations} bitwise=True", flush=True)

    graphs = {}
    for name in ("A", "B"):
        for _ in range(3):
            stages[name]["full"]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            stages[name]["full"]()
        graphs[name] = graph

    for replay in range(args.graph_replays):
        x.normal_()
        topk_weights.uniform_()
        if replay % 10 == 0:
            varied = synthetic_route(
                m,
                max(T, min(E, args.synthetic_active + (replay // 10) % 17 - 8)),
                20270000 + replay,
            ).to(device)
            inject_route(metadata, varied)
        graphs["A"].replay(); graphs["B"].replay()
        torch.cuda.synchronize()
        assert_same(states, f"graph replay={replay}")
    inject_route(metadata, topk)
    print(f"CORRECT graph_replays={args.graph_replays} bitwise=True", flush=True)

    samples = {stage: {"A": [], "B": []} for stage in STAGES}
    for round_index in range(args.rounds):
        order = ("A", "B", "B", "A") if round_index % 2 == 0 else ("B", "A", "A", "B")
        for name in order:
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
                f"trimmed_us={trimmed(values):.3f}",
                flush=True,
            )
    baseline_us = summary["full"]["A"]["trimmed_mean_us"]
    candidate_us = summary["full"]["B"]["trimmed_mean_us"]
    gain_pct = (baseline_us / candidate_us - 1.0) * 100.0
    saved_us = baseline_us - candidate_us
    decision = {
        "saved_us": saved_us,
        "gain_pct": gain_pct,
        "passes_22pct": candidate_us <= 0.78 * baseline_us,
        "passes_270us": saved_us >= 270.0,
    }
    report = {
        "shape": {"M": m, "T": T, "H": H, "I": I, "N": N},
        "route": route_summary,
        "route_mutation_repeat_fraction": [
            item["repeat_fraction"] for item in route_summaries
        ],
        "timings": summary,
        "decision": decision,
    }
    print(f"DECISION {json.dumps(decision, sort_keys=True)}", flush=True)
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(f"REPORT {args.output}", flush=True)


if __name__ == "__main__":
    main()
