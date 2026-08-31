#!/usr/bin/env python3
"""Oracle-only ABBA for paired DSpark gamma-one C128 sparse attention.

The replay rows must be the production M64 target-verify order
``[anchor_0, draft_0, ...]``.  The candidate launches one CTA for each adjacent
pair/split, reuses an LDS KV tile when both physical gather vectors match, and
writes the unchanged split2 workspace consumed by the production CK reducer.
It is not connected to any runtime selector.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.attention.dsv4.gfx90a_unified_sparse_decode import (
    run as run_baseline,
    workspace_size_bytes,
)
from sglang.kernels.ops.attention.dsv4.gfx90a_unified_sparse_pair_oracle import (
    preload as preload_pair,
    run as run_pair,
)


M = 64
H = 16
D = 512


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay",
        type=Path,
        default=Path(
            "/tmp/dsv4_pair_dump/"
            "layer_3_rank_0_c128_unified_sparse_m64.pt"
        ),
    )
    parser.add_argument("--physical-gpu", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--min-saving-us", type=float, default=15.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("warmup", "iterations", "rounds", "mutations", "graph_replays"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.rounds < 7:
        parser.error("at least seven ABBA rounds are required")
    if args.physical_gpu != 4:
        parser.error("this oracle is assigned to physical GPU 4")
    return args


def require_physical_gpu(expected: int) -> str:
    declared = []
    for name in ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        value = os.getenv(name)
        if value:
            declared.append((name, value))
    if not declared:
        raise RuntimeError(f"set HIP_VISIBLE_DEVICES={expected} before this oracle")
    for name, value in declared:
        if value.strip() != str(expected):
            raise RuntimeError(
                f"{name}={value!r}; expected exactly physical GPU {expected}"
            )
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("ROCm PyTorch and one visible GPU are required")
    torch.cuda.set_device(0)
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "")
    if arch.split(":", 1)[0] != "gfx90a":
        raise RuntimeError(f"expected gfx90a, got {arch!r}")
    return ",".join(f"{name}={value}" for name, value in declared)


def load_replay(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected dict replay")
    required = {
        "q",
        "unified_kv",
        "kv_indices",
        "kv_indptr",
        "attn_sink",
        "baseline_output",
        "positions",
        "softmax_scale",
        "provenance",
    }
    missing = required - payload.keys()
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")
    provenance = payload["provenance"]
    if provenance.get("compress_ratio") != 128:
        raise ValueError(f"{path}: expected C128 replay, got {provenance}")
    if tuple(payload["q"].shape) != (M, H, D):
        raise ValueError(f"{path}: expected q[M64,H16,D512]")
    if tuple(payload["kv_indptr"].shape) != (M + 1,):
        raise ValueError(f"{path}: expected kv_indptr[65]")
    if tuple(payload["positions"].shape) != (M,):
        raise ValueError(f"{path}: expected positions[64]")
    return payload


def validate_pair_contract(payload: dict) -> dict:
    indices = payload["kv_indices"].to(torch.int64).reshape(-1)
    indptr = payload["kv_indptr"].to(torch.int64).tolist()
    positions = payload["positions"].to(torch.int64).tolist()
    prefix_pairs = 0
    adjacent_pairs = 0
    shared = 0
    total = 0
    shared_tiles = 0
    split2_compatible_tiles = 0
    split2_saved_entries = 0
    for pair in range(M // 2):
        a_row, d_row = 2 * pair, 2 * pair + 1
        anchor = indices[indptr[a_row] : indptr[a_row + 1]]
        draft = indices[indptr[d_row] : indptr[d_row + 1]]
        if anchor.numel() <= draft.numel() and torch.equal(
            anchor, draft[: anchor.numel()]
        ):
            prefix_pairs += 1
        if positions[d_row] == positions[a_row] + 1:
            adjacent_pairs += 1
        common = min(anchor.numel(), draft.numel())
        equal = anchor[:common] == draft[:common]
        shared += int(equal.sum())
        total += anchor.numel() + draft.numel()
        # Exact lockstep 16-key tiles.  Split-boundary differences are handled
        # dynamically by the kernel and may reduce the realized count.
        shared_tiles += int(equal[: common - common % 16].view(-1, 16).all(1).sum())
        row_values = (anchor, draft)
        row_tiles = tuple((value.numel() + 15) // 16 for value in row_values)
        row_tiles_per_split = tuple((tiles + 1) // 2 for tiles in row_tiles)
        for split in range(2):
            first = tuple(split * count for count in row_tiles_per_split)
            last = tuple(
                min(row_tiles[row], first[row] + row_tiles_per_split[row])
                for row in range(2)
            )
            for step in range(max(last[row] - first[row] for row in range(2))):
                chunks = []
                for row in range(2):
                    tile = first[row] + step
                    start = tile * 16
                    chunks.append(
                        row_values[row][start : min(start + 16, row_values[row].numel())]
                        if tile < last[row]
                        else row_values[row][:0]
                    )
                common_chunk = min(chunks[0].numel(), chunks[1].numel())
                if torch.equal(
                    chunks[0][:common_chunk], chunks[1][:common_chunk]
                ):
                    split2_compatible_tiles += 1
                    split2_saved_entries += common_chunk
    if prefix_pairs != M // 2 or adjacent_pairs != M // 2:
        raise RuntimeError(
            "pair-query oracle requires every adjacent row pair to be an "
            f"anchor-prefix and adjacent-position pair; prefix={prefix_pairs}/32 "
            f"adjacent={adjacent_pairs}/32"
        )
    return {
        "prefix_pairs": prefix_pairs,
        "adjacent_pairs": adjacent_pairs,
        "shared_entries": shared,
        "total_entries": total,
        "set_ordered_load_reduction_upper_bound": shared / total,
        "lockstep_shared_tiles_before_split_partition": shared_tiles,
        "split2_prefix_compatible_tiles": split2_compatible_tiles,
        "split2_saved_entries": split2_saved_entries,
        "split2_load_reduction_upper_bound": split2_saved_entries / total,
    }


def capture(fn) -> torch.cuda.CUDAGraph:
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    torch.cuda.synchronize()
    return graph


def time_graph(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
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


def trimmed(values: list[float]) -> float:
    ordered = sorted(values)
    return statistics.fmean(ordered[1:-1])


def main() -> None:
    args = parse_args()
    placement = require_physical_gpu(args.physical_gpu)
    payload = load_replay(args.replay)
    contract = validate_pair_contract(payload)
    device = torch.device("cuda")
    q = payload["q"].to(device=device, dtype=torch.bfloat16).contiguous()
    kv = payload["unified_kv"].to(
        device=device, dtype=torch.bfloat16
    ).contiguous()
    indices = payload["kv_indices"].to(device=device, dtype=torch.int32).contiguous()
    indptr = payload["kv_indptr"].to(device=device, dtype=torch.int32).contiguous()
    sink = payload["attn_sink"].to(
        device=device, dtype=torch.float32
    ).contiguous()
    production = payload["baseline_output"].to(
        device=device, dtype=torch.bfloat16
    ).contiguous()
    scale = float(payload["softmax_scale"])

    baseline_out = torch.empty_like(q)
    pair_out = torch.empty_like(q)
    workspace_bytes = workspace_size_bytes(tokens=M)
    baseline_workspace = torch.empty(
        workspace_bytes, dtype=torch.uint8, device=device
    )
    pair_workspace = torch.empty_like(baseline_workspace)

    def baseline() -> None:
        run_baseline(
            q,
            kv,
            indices,
            indptr,
            sink,
            baseline_out,
            baseline_workspace,
            scale,
        )

    def pair() -> None:
        run_pair(
            q,
            kv,
            indices,
            indptr,
            sink,
            pair_out,
            pair_workspace,
            scale,
            128,
        )

    # Compile both modules before any graph capture.
    baseline()
    preload_pair()
    pair()
    torch.cuda.synchronize()
    if not torch.equal(baseline_out, pair_out):
        delta = (baseline_out.float() - pair_out.float()).abs()
        raise RuntimeError(
            "initial pair output is not bitwise exact: "
            f"max_abs={delta.max().item()} nonzero={torch.count_nonzero(delta).item()}"
        )

    production_delta = (production.float() - pair_out.float()).abs()
    production_max_abs = float(production_delta.max())
    production_rel_l2 = float(
        torch.linalg.vector_norm(production.float() - pair_out.float())
        / torch.linalg.vector_norm(production.float()).clamp_min(1.0e-12)
    )

    mutation_max_abs = 0.0
    generator = torch.Generator(device=device).manual_seed(20260831)
    for mutation in range(args.mutations):
        q.normal_(generator=generator)
        sink.normal_(generator=generator)
        # Exercise the shared loads without paying a full compact-KV rewrite on
        # every mutation.  Ten independent KV states still cover all 100 Q/sink
        # mutations required by the gate.
        if mutation % 10 == 0:
            kv.normal_(generator=generator)
        baseline()
        pair()
        torch.cuda.synchronize()
        if not torch.equal(baseline_out, pair_out):
            delta = (baseline_out.float() - pair_out.float()).abs()
            mutation_max_abs = max(mutation_max_abs, float(delta.max()))
            raise RuntimeError(
                f"mutation={mutation} pair mismatch max_abs={delta.max().item()} "
                f"nonzero={torch.count_nonzero(delta).item()}"
            )

    baseline_graph = capture(baseline)
    pair_graph = capture(pair)
    q.normal_(generator=generator)
    kv.normal_(generator=generator)
    sink.normal_(generator=generator)
    pair()
    torch.cuda.synchronize()
    graph_expected = pair_out.clone()
    for _ in range(args.graph_replays):
        pair_graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(graph_expected, pair_out):
        delta = (graph_expected.float() - pair_out.float()).abs()
        raise RuntimeError(
            f"graph replay changed pair output max_abs={delta.max().item()}"
        )
    baseline()
    pair()
    torch.cuda.synchronize()
    if not torch.equal(baseline_out, pair_out):
        raise RuntimeError("post-graph pair output differs from baseline")

    timings = {"baseline": [], "pair": []}
    for _ in range(args.rounds):
        for provider in ("baseline", "pair", "pair", "baseline"):
            graph = baseline_graph if provider == "baseline" else pair_graph
            timings[provider].append(
                time_graph(graph, args.warmup, args.iterations)
            )
    baseline_us = trimmed(timings["baseline"])
    pair_us = trimmed(timings["pair"])
    saving_us = baseline_us - pair_us
    result = {
        "schema": "dsv4_dspark_m64_pair_ck_sparse_oracle_v1",
        "placement": placement,
        "replay": str(args.replay),
        "provenance": payload["provenance"],
        "contract": contract,
        "correctness": {
            "mutations": args.mutations,
            "pair_vs_ck_bitwise_exact": True,
            "mutation_max_abs": mutation_max_abs,
            "graph_replays": args.graph_replays,
            "graph_bitwise_exact": True,
            "production_max_abs": production_max_abs,
            "production_rel_l2": production_rel_l2,
        },
        "timing": {
            "baseline_samples_us": timings["baseline"],
            "pair_samples_us": timings["pair"],
            "baseline_trimmed_us": baseline_us,
            "pair_trimmed_us": pair_us,
            "saving_us": saving_us,
            "gain_pct": (baseline_us / pair_us - 1.0) * 100.0,
            "continuation_gate_us": args.min_saving_us,
            "passes_gate": saving_us >= args.min_saving_us,
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    if saving_us < args.min_saving_us:
        raise RuntimeError(
            f"pair-query saving {saving_us:.3f}us misses "
            f"{args.min_saving_us:.3f}us continuation gate"
        )


if __name__ == "__main__":
    main()
