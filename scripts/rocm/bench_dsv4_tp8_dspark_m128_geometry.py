#!/usr/bin/env python3
"""TP8 I256 M128 routed-FP4 geometry component screen (not service timing).

Compares the production prefill-style M128 geometry (G416/D312) with the
candidate decode-style geometry G832/D832. This is not a TP8 E2E result.
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
    _jit_gate_up_grouped_row_prefetch,
)
from sglang.kernels.ops.quantization.int8_kernel import (
    _per_token_group_quant_int8,
)
from sglang.kernels.jit.utils import load_jit, make_cpp_args


E, T, H, I, N = 256, 6, 4096, 256, 4096
ASSIGNMENTS, ROWS, WAVES, LDS_LUT = 4, 2, 8, 2


def tp8_down_row_prefetch(
    tokens: int, assignments: int, waves: int, blocks: int
):
    """Load the subgroup-8 oracle; the public helper is TP4/subgroup-16 only."""
    shape = (E, tokens, T, N, I, assignments, waves, blocks, 2)
    cpp = make_cpp_args(*shape)
    return load_jit(
        "gfx90a_fp4_tp8_down_prefetch_oracle",
        *shape,
        cuda_files=["deepseek_v4/gfx90a_fp4_tp8_down_prefetch_oracle.cuh"],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{cpp}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{cpp}>::reduce",
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


def load_real_counts(
    path: Path, layer: int, tokens: int, *, stat_pass: int, world_size: int
) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = payload.get("records")
    if records is None and "logical_count" in payload:
        counts = payload["logical_count"][:, layer].to(torch.int64)
        expected = tokens * T * world_size
        matches = counts[counts.sum(dim=-1) == expected]
        if not matches.numel():
            raise RuntimeError(
                f"no TP-summed M{tokens}/top-{T} stat record in {path}"
            )
        selected = matches[stat_pass % matches.shape[0]]
        if torch.any(selected % world_size):
            raise RuntimeError("TP-summed expert counts are not world-size divisible")
        return selected // world_size
    if not isinstance(records, list):
        raise RuntimeError("expected a per-pass or stat expert recorder")
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
    return statistics.fmean(values) if len(values) <= 2 else statistics.fmean(sorted(values)[1:-1])


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
    parser.add_argument("--recorder", type=Path)
    parser.add_argument("--distribution", choices=("balanced","skewed"), default="balanced")
    parser.add_argument("--tokens", type=int, choices=(64, 96, 128), default=128)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--stat-pass", type=int, default=-1)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument(
        "--no-lds-lut",
        action="store_true",
        help="match the production TP8 DSpark M128 non-LDS lookup path",
    )
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
    parser.add_argument(
        "--screen-row-prefetch",
        action="store_true",
        help=(
            "compare the existing LDS row-prefetch gate/down oracle with the "
            "current G832/D832 path; component-only and never a service selector"
        ),
    )
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="time gate, intermediate quant, down partial and reducer separately",
    )
    parser.add_argument(
        "--screen-prefetch-grid",
        action="store_true",
        help="scan bounded gate/down CTA counts for the row-prefetch tactic",
    )
    args = parser.parse_args()
    if args.mutations < 100 or args.graph_replays < 1000 or args.rounds != 1:
        raise ValueError("formal oracle requires 100 mutations, 1000 replays, one ABBA")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("this oracle requires gfx90a")

    tokens = args.tokens
    if args.recorder:
        counts = load_real_counts(
            args.recorder,
            args.layer,
            tokens,
            stat_pass=args.stat_pass,
            world_size=args.world_size,
        )
        topk_ids = reconstruct_topk_from_counts(counts, m=tokens, topk=T).cuda()
    else:
        torch.manual_seed(20260909)
        probabilities = torch.ones(E)
        if args.distribution == "skewed":
            probabilities[:32] = 16
        topk_ids = torch.stack([torch.multinomial(probabilities,T,replacement=False) for _ in range(tokens)]).to(torch.int32).cuda()
        counts = torch.bincount(topk_ids.cpu().long().flatten(),minlength=E)
        print("SYNTHETIC",args.distribution,"not a real-route E2E oracle",flush=True)
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
    s13 = torch.randint(123,128,(E, 2 * I, H // 32), dtype=torch.uint8, device="cuda")
    w2 = torch.randint(0, 256, (E, N, I // 2), dtype=torch.uint8, device="cuda")
    s2 = torch.randint(123,128,(E, N, I // 32), dtype=torch.uint8, device="cuda")

    lds_lut = False if args.no_lds_lut else LDS_LUT
    geometries = {
        "prefill": (4, 2, 8, 416, 312),
        "decode": (4, 2, 8, 832, 832),
    }
    if args.screen_grid:
        geometries.update(
            {
                "g624_d832": (4, 2, 8, 624, 832),
                "g1040_d832": (4, 2, 8, 1040, 832),
                "g832_d624": (4, 2, 8, 832, 624),
                "g832_d1040": (4, 2, 8, 832, 1040),
                "g832_d1248": (4, 2, 8, 832, 1248),
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
    if args.screen_row_prefetch:
        geometries["row_prefetch"] = (4, 2, 8, 832, 832)
    if args.screen_prefetch_grid:
        if not args.screen_row_prefetch:
            raise ValueError("--screen-prefetch-grid requires --screen-row-prefetch")
        geometries.update(
            {
                "row_prefetch_g624_d832": (4, 2, 8, 624, 832),
                "row_prefetch_g1040_d832": (4, 2, 8, 1040, 832),
                "row_prefetch_g1248_d832": (4, 2, 8, 1248, 832),
                "row_prefetch_g832_d624": (4, 2, 8, 832, 624),
                "row_prefetch_g832_d1040": (4, 2, 8, 832, 1040),
                "row_prefetch_g832_d1248": (4, 2, 8, 832, 1248),
            }
        )
    states: dict[str, dict[str, torch.Tensor]] = {}
    runs = {}
    modules = {}
    for name, (
        assignments,
        rows,
        waves,
        gate_blocks,
        down_blocks,
    ) in geometries.items():
        metadata = metadata_a4 if assignments == 4 else metadata_a8
        if name.startswith("row_prefetch"):
            # The existing row-prefetch kernels stage decoded FP4 through LDS.
            # Keep this an explicit oracle rather than silently conflating it
            # with the production non-LDS lookup mode.
            gate = _jit_gate_up_grouped_row_prefetch(
                E, tokens, T, I, H, assignments, rows, waves, gate_blocks, 2
            )
            down = tp8_down_row_prefetch(tokens, assignments, waves, down_blocks)
        else:
            gate = _jit_gate_up_grouped(
                E, tokens, T, I, H, assignments, rows, waves, gate_blocks, lds_lut
            )
            down = _jit_down_grouped(
                E, tokens, T, N, I, assignments, rows, waves, down_blocks, lds_lut
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
        modules[name] = (gate, down, metadata)

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
            elapsed = time_graph(graphs[name], args.iterations)
            values[name].append(elapsed)
            print("ABBA_SAMPLE", name, elapsed, flush=True)
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
    if args.screen_row_prefetch:
        candidate_us = trimmed(values["row_prefetch"])
        print(
            f"PREFETCH_RESULT tokens={tokens} current_us={decode_us:.3f} "
            f"candidate_us={candidate_us:.3f} "
            f"saving_us={decode_us - candidate_us:.3f} "
            f"gain_pct={(decode_us / candidate_us - 1.0) * 100.0:.3f}",
            flush=True,
        )
    if args.screen_prefetch_grid:
        reference_us = trimmed(values["row_prefetch"])
        for name in geometries:
            if not name.startswith("row_prefetch_"):
                continue
            candidate_us = trimmed(values[name])
            print(
                f"PREFETCH_GRID_RESULT name={name} reference_us={reference_us:.3f} "
                f"candidate_us={candidate_us:.3f} "
                f"saving_us={reference_us - candidate_us:.3f} "
                f"gain_pct={(reference_us / candidate_us - 1.0) * 100.0:.3f}",
                flush=True,
            )
    if args.breakdown:
        for name in ("decode", "row_prefetch"):
            if name not in modules:
                continue
            gate, down, metadata = modules[name]
            state = states[name]
            stage_runs = {
                "gate": lambda gate=gate, metadata=metadata, state=state: gate.run(
                    xq, xscale, w13, s13, metadata.sorted_ids,
                    metadata.sorted_experts, metadata.valid,
                    state["intermediate"], 10.0
                ),
                "intermediate_quant": lambda state=state: quant_into(
                    state["intermediate"], state["iq"], state["iscale"]
                ),
                "down_partial": lambda down=down, metadata=metadata, state=state: down.run_partial(
                    state["iq"], state["iscale"], w2, s2,
                    metadata.sorted_ids, metadata.sorted_experts,
                    metadata.valid, topk_weights, state["partial"]
                ),
                "reducer": lambda down=down, state=state: down.reduce(
                    state["partial"], state["output"]
                ),
            }
            stage_graphs = {}
            for stage, stage_run in stage_runs.items():
                stage_run()
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    stage_run()
                stage_graphs[stage] = graph
            stage_values = {stage: [] for stage in stage_graphs}
            for order in (
                tuple(stage_graphs),
                tuple(reversed(stage_graphs)),
                tuple(stage_graphs),
                tuple(reversed(stage_graphs)),
            ):
                for stage in order:
                    stage_values[stage].append(time_graph(stage_graphs[stage], args.iterations))
            print(
                "BREAKDOWN",
                name,
                {stage: round(trimmed(samples), 3)
                 for stage, samples in stage_values.items()},
                flush=True,
            )


if __name__ == "__main__":
    main()
