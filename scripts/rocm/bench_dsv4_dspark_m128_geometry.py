#!/usr/bin/env python3
"""Formal M64/M96/M128 routed-FP4 geometry oracle for DSpark verification.

Compares the production prefill-style M128 geometry (G416/D312) with the
decode-style geometry used by the successful M64--M96 family (G2080/D832).
Both arms use A4/R2, the same LDS E2M1 lookup, group-32 INT8 activation
quantization and the fixed-order FP32 output reduction.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    _per_token_group_quant_int8,
)


E, T, H, I, N = 256, 6, 4096, 512, 4096
ASSIGNMENTS, ROWS, WAVES, LDS_LUT = 4, 2, 8, 2


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


def load_real_counts(path: Path, layer: int, tokens: int) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if not isinstance(records, list):
        raise RuntimeError("expected a per-pass recorder with a records list")
    matches = [
        record["global_physical_count"][layer].to(torch.int64)
        for record in records
        if int(record["global_physical_count"][layer].sum()) == tokens * T
    ]
    if not matches:
        raise RuntimeError(f"no M{tokens}/top-{T} target record in {path}")
    # Pick the middle full-BS record rather than a capture/startup boundary.
    return matches[len(matches) // 2]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recorder", type=Path, required=True)
    parser.add_argument("--tokens", type=int, choices=(64, 96, 128), default=128)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument(
        "--screen-a8",
        action="store_true",
        help="also compare A8/R1/W4 decode geometries against production A4",
    )
    parser.add_argument(
        "--screen-grid",
        action="store_true",
        help="scan independent A4 gate/down grids around the production geometry",
    )
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 7:
        raise ValueError("formal oracle requires 100 mutations, 1000 replays, 7 rounds")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("this oracle requires gfx90a")

    tokens = args.tokens
    counts = load_real_counts(args.recorder, args.layer, tokens)
    topk_ids = reconstruct_topk_from_counts(counts, m=tokens, topk=T).cuda()
    metadata_a4 = make_metadata(topk_ids, assignments=ASSIGNMENTS)
    metadata_a8 = make_metadata(topk_ids, assignments=8)
    print(
        f"ROUTE layer={args.layer} active={int((counts > 0).sum())} "
        f"a4_scans={metadata_a4.sorted_experts.numel()} "
        f"a8_scans={metadata_a8.sorted_experts.numel()} "
        f"a4_valid={metadata_a4.valid.cpu().tolist()} "
        f"a8_valid={metadata_a8.valid.cpu().tolist()} "
        f"max_occ={int(counts.max())}",
        flush=True,
    )

    torch.manual_seed(20260831)
    x = torch.randn((tokens, H), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((tokens, H), dtype=torch.int8, device="cuda")
    xscale = torch.empty((tokens, H // 32), dtype=torch.float32, device="cuda")
    topk_weights = torch.rand((tokens, T), dtype=torch.float32, device="cuda")
    w13 = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s13 = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.full((E, N, I // 32), 127, dtype=torch.uint8, device="cuda")

    geometries = {
        "prefill": (4, 2, 8, 416, 312),
        "decode": (4, 2, 8, 2080, 832),
    }
    if args.screen_grid:
        geometries.update(
            {
                "g1248_d832": (4, 2, 8, 1248, 832),
                "g1664_d832": (4, 2, 8, 1664, 832),
                "g2496_d832": (4, 2, 8, 2496, 832),
                "g3120_d832": (4, 2, 8, 3120, 832),
                "g2080_d624": (4, 2, 8, 2080, 624),
                "g2080_d1040": (4, 2, 8, 2080, 1040),
                "g2080_d1248": (4, 2, 8, 2080, 1248),
                "w4_g2080_d832": (4, 2, 4, 2080, 832),
            }
        )
    if args.screen_a8:
        geometries.update(
            {
                "a8_g1664_d832": (8, 1, 4, 1664, 832),
                "a8_g2080_d832": (8, 1, 4, 2080, 832),
                "a8_g1664_d1248": (8, 1, 4, 1664, 1248),
                "a8_g2080_d1248": (8, 1, 4, 2080, 1248),
            }
        )
    states: dict[str, dict[str, torch.Tensor]] = {}
    runs = {}
    for name, (
        assignments,
        rows,
        waves,
        gate_blocks,
        down_blocks,
    ) in geometries.items():
        metadata = metadata_a4 if assignments == 4 else metadata_a8
        gate = _jit_gate_up_grouped(
            E, tokens, T, I, H, assignments, rows, waves, gate_blocks, LDS_LUT
        )
        down = _jit_down_grouped(
            E, tokens, T, N, I, assignments, rows, waves, down_blocks, LDS_LUT
        )
        state = {
            "intermediate": torch.zeros(
                (tokens, T, I), dtype=torch.bfloat16, device="cuda"
            ),
            "iq": torch.zeros((tokens, T, I), dtype=torch.int8, device="cuda"),
            "iscale": torch.zeros(
                (tokens, T, I // 32), dtype=torch.float32, device="cuda"
            ),
            "partial": torch.zeros(
                (tokens, T, N), dtype=torch.float32, device="cuda"
            ),
            "output": torch.zeros(
                (tokens, N), dtype=torch.bfloat16, device="cuda"
            ),
        }

        def run(gate=gate, down=down, state=state, metadata=metadata) -> None:
            gate.run(
                xq,
                xscale,
                w13,
                s13,
                metadata.sorted_ids,
                metadata.sorted_experts,
                metadata.valid,
                state["intermediate"],
                10.0,
            )
            quant_into(state["intermediate"], state["iq"], state["iscale"])
            down.run_partial(
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
            down.reduce(state["partial"], state["output"])

        states[name] = state
        runs[name] = run

    def assert_exact(label: str) -> None:
        for tensor_name in states["prefill"]:
            expected = states["prefill"][tensor_name]
            for name, state in states.items():
                actual = state[tensor_name]
                if not torch.equal(expected, actual):
                    delta = (expected.float() - actual.float()).abs()
                    mismatch = torch.nonzero(delta, as_tuple=False)
                    raise RuntimeError(
                        f"{label} {name} {tensor_name} mismatch "
                        f"max_abs={delta.max().item()} count={mismatch.shape[0]} "
                        f"first={mismatch[:8].cpu().tolist()}"
                    )

    for mutation in range(args.mutations):
        x.normal_()
        topk_weights.uniform_()
        quant_into(x, xq, xscale)
        runs["prefill"]()
        runs["decode"]()
        for name in states:
            if name not in ("prefill", "decode"):
                runs[name]()
        torch.cuda.synchronize()
        assert_exact(f"mutation={mutation}")
    print(f"CORRECT mutations={args.mutations} bitwise_exact=True", flush=True)

    graphs = {}
    for name in geometries:
        for _ in range(10):
            runs[name]()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runs[name]()
        graph.replay()
        torch.cuda.synchronize()
        expected = states[name]["output"].clone()
        for _ in range(args.graph_replays):
            graph.replay()
        torch.cuda.synchronize()
        if not torch.equal(expected, states[name]["output"]):
            raise RuntimeError(f"{name} graph replay changed output")
        graphs[name] = graph
    print(f"GRAPH replays={args.graph_replays} bitwise_exact=True", flush=True)

    values = {name: [] for name in graphs}
    profile_order = list(geometries)
    for round_idx in range(args.rounds):
        forward = profile_order if round_idx % 2 == 0 else list(reversed(profile_order))
        for name in (*forward, *reversed(forward)):
            values[name].append(time_graph(graphs[name], args.iterations))
    prefill_us = trimmed(values["prefill"])
    decode_us = trimmed(values["decode"])
    print(
        f"RESULT tokens={tokens} prefill_us={prefill_us:.3f} decode_us={decode_us:.3f} "
        f"saving_us={prefill_us - decode_us:.3f} "
        f"gain_pct={(prefill_us / decode_us - 1.0) * 100.0:.3f}",
        flush=True,
    )
    if args.screen_grid:
        for name in geometries:
            if name in ("prefill", "decode") or name.startswith("a8_"):
                continue
            candidate_us = trimmed(values[name])
            print(
                f"GRID_RESULT tokens={tokens} name={name} "
                f"current_us={decode_us:.3f} candidate_us={candidate_us:.3f} "
                f"saving_us={decode_us - candidate_us:.3f} "
                f"gain_pct={(decode_us / candidate_us - 1.0) * 100.0:.3f}",
                flush=True,
            )
    if args.screen_a8:
        for name in geometries:
            if not name.startswith("a8_"):
                continue
            candidate_us = trimmed(values[name])
            print(
                f"A8_RESULT name={name} current_a4_us={decode_us:.3f} "
                f"candidate_us={candidate_us:.3f} "
                f"saving_us={decode_us - candidate_us:.3f} "
                f"gain_pct={(decode_us / candidate_us - 1.0) * 100.0:.3f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
