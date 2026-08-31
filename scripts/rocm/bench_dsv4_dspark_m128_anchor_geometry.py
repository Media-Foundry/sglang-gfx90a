#!/usr/bin/env python3
"""Screen M128 kernel tactics and physical M32 anchor compaction.

The service checkpoint masks three DSpark draft rows with ``-1`` but retains
the physical M128 routed-MoE tensor.  A previous M32 compaction experiment was
slower because it also changed the selected kernel family.  This oracle keeps
the real anchor route fixed and separates three possibilities:

* ``sentinel_m128``: production M128 non-DPP grouped kernels (A);
* ``sentinel_m128_dpp_gate``: M128 DPP gate only (G);
* ``sentinel_m128_row_prefetch``: M128 logical-scale down only (L);
* ``sentinel_m128_dpp_row_prefetch``: both M128 tactics (B);
* ``sentinel_m128_dpp_row_prefetch_w4``: both tactics with four down waves
  (B4);
* ``compact_m32_standard``: physical M32 with the same non-DPP family;
* ``compact_m32_dpp``: physical M32 with the profile's DPP gate and logical-
  scale row-prefetch down kernel.

The M32 arms include anchor gather and full-layout zero/scatter costs.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_dspark_m128_geometry import quant_into
from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import make_metadata
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_down_grouped_row_prefetch_logical_scale,
    _jit_gate_up_grouped,
    _jit_gate_up_grouped_dpp,
)


E, M_FULL, M_ANCHOR, T, H, I, N = 256, 128, 32, 6, 4096, 512, 4096
A, R, W, GATE_BLOCKS, DOWN_BLOCKS, LDS = 4, 2, 8, 2080, 832, 2


def trimmed(values: list[float]) -> float:
    return statistics.fmean(sorted(values)[1:-1])


def time_graph(graph: torch.cuda.CUDAGraph, iterations: int) -> float:
    for _ in range(10):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def load_real_m128_topk(path: Path, layer: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    matches = []
    for record in payload.get("records", []):
        topk = record.get("topk_ids_of_layer")
        input_ids = record.get("input_ids")
        n = len(input_ids) if isinstance(input_ids, list) else getattr(input_ids, "numel", lambda: 0)()
        if isinstance(topk, torch.Tensor) and n == M_FULL:
            candidate = topk[layer].to(torch.int32).contiguous()
            if candidate.shape == (M_FULL, T) and bool((candidate >= 0).all()):
                matches.append(candidate)
    if not matches:
        raise RuntimeError(f"no full M128 per-token route in {path}")
    return matches[len(matches) // 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 7:
        raise ValueError("formal oracle requires 100 mutations, 1000 replays, 7 rounds")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("this oracle requires gfx90a")

    real_topk = load_real_m128_topk(args.recorder, args.layer)
    anchor_topk = real_topk[0::4].contiguous().cuda()
    sentinel_topk = real_topk.clone()
    sentinel_topk.view(32, 4, T)[:, 1:].fill_(-1)
    sentinel_topk = sentinel_topk.cuda()
    meta_full = make_metadata(sentinel_topk, assignments=A)
    meta_anchor = make_metadata(anchor_topk, assignments=A)
    print(
        f"ROUTE layer={args.layer} anchors={M_ANCHOR} "
        f"active={int(torch.unique(anchor_topk).numel())} "
        f"a4_scans={meta_anchor.sorted_experts.numel()} "
        f"valid={meta_anchor.valid.cpu().tolist()}",
        flush=True,
    )

    torch.manual_seed(20260901)
    x = torch.randn((M_FULL, H), dtype=torch.bfloat16, device="cuda")
    topk_weights = torch.rand((M_FULL, T), dtype=torch.float32, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device="cuda")

    specs = {
        "sentinel_m128": (M_FULL, meta_full, "standard", "standard", W),
        "sentinel_m128_dpp_gate": (M_FULL, meta_full, "dpp", "standard", W),
        "sentinel_m128_row_prefetch": (
            M_FULL, meta_full, "standard", "row_prefetch", W
        ),
        "sentinel_m128_dpp_row_prefetch": (
            M_FULL, meta_full, "dpp", "row_prefetch", W
        ),
        "sentinel_m128_dpp_row_prefetch_w4": (
            M_FULL, meta_full, "dpp", "row_prefetch", 4
        ),
        "compact_m32_standard": (
            M_ANCHOR, meta_anchor, "standard", "standard", W
        ),
        "compact_m32_dpp": (
            M_ANCHOR, meta_anchor, "dpp", "row_prefetch", W
        ),
    }
    states = {}
    runs = {}
    for name, (m, metadata, gate_kind, down_kind, down_waves) in specs.items():
        gate = (
            _jit_gate_up_grouped_dpp(E, m, T, I, H, A, R, W, GATE_BLOCKS, LDS)
            if gate_kind == "dpp"
            else _jit_gate_up_grouped(E, m, T, I, H, A, R, W, GATE_BLOCKS, LDS)
        )
        down = (
            _jit_down_grouped_row_prefetch_logical_scale(
                E, m, T, N, I, A, down_waves, DOWN_BLOCKS, LDS
            )
            if down_kind == "row_prefetch"
            else _jit_down_grouped(
                E, m, T, N, I, A, R, down_waves, DOWN_BLOCKS, LDS
            )
        )
        state = {
            "x": torch.empty((m, H), dtype=torch.bfloat16, device="cuda"),
            "weights": torch.empty((m, T), dtype=torch.float32, device="cuda"),
            "xq": torch.empty((m, H), dtype=torch.int8, device="cuda"),
            "xscale": torch.empty((m, H // 32), dtype=torch.float32, device="cuda"),
            "intermediate": torch.zeros((m, T, I), dtype=torch.bfloat16, device="cuda"),
            "iq": torch.zeros((m, T, I), dtype=torch.int8, device="cuda"),
            "iscale": torch.zeros((m, T, I // 32), dtype=torch.float32, device="cuda"),
            "partial": torch.zeros((m, T, N), dtype=torch.float32, device="cuda"),
            "output": torch.zeros((m, N), dtype=torch.bfloat16, device="cuda"),
            "full_output": torch.zeros((M_FULL, N), dtype=torch.bfloat16, device="cuda"),
        }

        def run(
            m=m,
            metadata=metadata,
            gate=gate,
            down=down,
            state=state,
        ) -> None:
            if m == M_FULL:
                state["x"].copy_(x)
                state["weights"].copy_(topk_weights)
            else:
                state["x"].copy_(x[0::4])
                state["weights"].copy_(topk_weights[0::4])
            quant_into(state["x"], state["xq"], state["xscale"])
            gate.run(
                state["xq"], state["xscale"], w13, s13,
                metadata.sorted_ids, metadata.sorted_experts, metadata.valid,
                state["intermediate"], 10.0,
            )
            quant_into(state["intermediate"], state["iq"], state["iscale"])
            down.run_partial(
                state["iq"], state["iscale"], w2, s2,
                metadata.sorted_ids, metadata.sorted_experts, metadata.valid,
                state["weights"], state["partial"],
            )
            down.reduce(state["partial"], state["output"])
            if m == M_FULL:
                state["full_output"].copy_(state["output"])
            else:
                state["full_output"].zero_()
                state["full_output"][0::4].copy_(state["output"])

        states[name] = state
        runs[name] = run

    exact = {name: True for name in specs}
    for mutation in range(args.mutations):
        x.normal_()
        topk_weights.uniform_()
        for run in runs.values():
            run()
        torch.cuda.synchronize()
        reference = states["sentinel_m128"]["full_output"]
        for name, state in states.items():
            if not torch.equal(reference, state["full_output"]):
                exact[name] = False
                delta = (reference.float() - state["full_output"].float()).abs()
                raise RuntimeError(
                    f"mutation={mutation} {name} mismatch "
                    f"max_abs={delta.max().item()} "
                    f"count={int(torch.count_nonzero(delta))}"
                )
    print(f"CORRECT mutations={args.mutations} exact={exact}", flush=True)

    graphs = {}
    for name, run in runs.items():
        for _ in range(10):
            run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            run()
        expected = states[name]["full_output"].clone()
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, states[name]["full_output"]):
            raise RuntimeError(f"{name} graph replay changed output")
        graphs[name] = graph
    print(f"GRAPH replays={args.graph_replays} bitwise_exact=True", flush=True)

    values = {name: [] for name in graphs}
    names = list(graphs)
    for round_idx in range(args.rounds):
        order = names if round_idx % 2 == 0 else list(reversed(names))
        for name in (*order, *reversed(order)):
            values[name].append(time_graph(graphs[name], args.iterations))
    result = {
        "samples_us": values,
        "median_us": {name: statistics.median(v) for name, v in values.items()},
        "trimmed_us": {name: trimmed(v) for name, v in values.items()},
        "exact": exact,
    }
    base = result["trimmed_us"]["sentinel_m128"]
    result["gain_pct_vs_sentinel"] = {
        name: (base / value - 1.0) * 100.0
        for name, value in result["trimmed_us"].items()
    }
    print(json.dumps(result, indent=2), flush=True)
    if args.output:
        args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
