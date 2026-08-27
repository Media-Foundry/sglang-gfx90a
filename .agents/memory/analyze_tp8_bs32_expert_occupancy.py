#!/usr/bin/env python3
"""Analyze TP8 BS32 DeepSeek-V4 expert occupancy recorder dumps.

The ``stat`` recorder stores the logical expert count after a TP-wide sum.  For
the TP8 profile that count must be divided by eight.  This tool selects only
complete BS32 decode passes, drops an explicit warm-decode prefix, and emits
machine-readable CSV and JSON summaries for hash-router and learned-router
layers separately.

The reported byte counts are *logical weight bytes requested by the grouped
kernel*.  They intentionally do not claim physical HBM traffic because L2
reuse and cache-line effects require hardware counters.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from typing import Iterable

import torch


OCCUPANCY_BUCKETS = (
    ("occ1", 1, 1),
    ("occ2", 2, 2),
    ("occ3_4", 3, 4),
    ("occ5_8", 5, 8),
    ("occ9_16", 9, 16),
    ("occ17_32", 17, 32),
    ("occ33_plus", 33, None),
)
TILES = (1, 2, 4, 8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recorder",
        nargs="?",
        help="expert_distribution_recorder_*.pt (defaults to newest by mtime)",
    )
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--num-hash-layers", type=int, default=3)
    parser.add_argument(
        "--warmup-passes",
        type=int,
        default=32,
        help="complete BS32 decode passes to discard before the window",
    )
    parser.add_argument(
        "--window-passes", type=int, default=128, help="complete passes to analyze"
    )
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument(
        "--intermediate-per-rank",
        type=int,
        default=256,
        help="TP-local routed-expert intermediate width",
    )
    parser.add_argument("--fp4-group-size", type=int, default=32)
    parser.add_argument("--csv", type=Path, help="write summary rows as CSV")
    parser.add_argument("--json", type=Path, help="write full summary as JSON")
    parser.add_argument(
        "--include-layers",
        action="store_true",
        help="also emit one summary row per layer",
    )
    return parser.parse_args()


def newest_dump() -> str:
    candidates = glob.glob("/tmp/expert_distribution_recorder_*.pt")
    if not candidates:
        raise FileNotFoundError("no /tmp expert distribution recorder dump found")
    return max(candidates, key=lambda path: Path(path).stat().st_mtime_ns)


def logical_weight_bytes_per_scan(hidden: int, intermediate: int, group: int) -> int:
    if hidden % group or intermediate % group:
        raise ValueError("hidden and intermediate dimensions must divide fp4 group size")
    # w13: [2I, H] packed FP4 plus one E8M0 byte per group-32.
    gate_up = (2 * intermediate * hidden) // 2 + 2 * intermediate * (hidden // group)
    # w2: [H, I] packed FP4 plus one E8M0 byte per group-32.
    down = (hidden * intermediate) // 2 + hidden * (intermediate // group)
    return gate_up + down


def summarize(
    values: torch.Tensor,
    *,
    scope: str,
    passes: int,
    layers: int,
    weight_bytes_per_scan: int,
) -> dict[str, float | int | str]:
    """Summarize a [passes, layers, experts] occupancy tensor."""
    flat = values.reshape(-1)
    nonzero = flat[flat > 0]
    samples = passes * layers
    total_assignments = int(nonzero.sum())
    total_active = int(nonzero.numel())
    row: dict[str, float | int | str] = {
        "scope": scope,
        "passes": passes,
        "layers": layers,
        "pass_layer_samples": samples,
        "assignments_per_layer": total_assignments / samples,
        "active_experts_mean": total_active / samples,
        "max_occupancy_mean": values.max(dim=-1).values.float().mean().item(),
        "max_occupancy_p50": torch.quantile(
            values.max(dim=-1).values.float(), 0.50
        ).item(),
        "max_occupancy_p95": torch.quantile(
            values.max(dim=-1).values.float(), 0.95
        ).item(),
        "weight_bytes_per_scan": weight_bytes_per_scan,
    }

    for name, lower, upper in OCCUPANCY_BUCKETS:
        mask = flat >= lower
        if upper is not None:
            mask &= flat <= upper
        expert_count = int(mask.sum())
        assignment_count = int(flat[mask].sum())
        row[f"{name}_experts_mean"] = expert_count / samples
        row[f"{name}_expert_pct"] = (
            100.0 * expert_count / total_active if total_active else 0.0
        )
        row[f"{name}_assignment_pct"] = (
            100.0 * assignment_count / total_assignments if total_assignments else 0.0
        )

    for tile in TILES:
        scans = int(torch.div(nonzero + tile - 1, tile, rounding_mode="floor").sum())
        capacity = scans * tile
        padding = capacity - total_assignments
        prefix = f"A{tile}"
        row[f"{prefix}_scans_mean"] = scans / samples
        row[f"{prefix}_padding_mean"] = padding / samples
        row[f"{prefix}_util_pct"] = (
            100.0 * total_assignments / capacity if capacity else 0.0
        )
        row[f"{prefix}_estimated_weight_bytes_mean"] = (
            scans * weight_bytes_per_scan / samples
        )
        row[f"{prefix}_estimated_weight_bytes_per_useful_assignment"] = (
            scans * weight_bytes_per_scan / total_assignments
        )

    # A1 for occupancy 1, A2 for 2, A4 for everything else.  This has the
    # same number of weight scans as fixed A4; it only removes padded lanes.
    adaptive_capacity = torch.where(
        nonzero == 1,
        nonzero,
        torch.where(
            nonzero == 2,
            nonzero,
            4 * torch.div(nonzero + 3, 4, rounding_mode="floor"),
        ),
    )
    adaptive_padding = int(adaptive_capacity.sum()) - total_assignments
    row["A1_A2_A4_scans_mean"] = row["A4_scans_mean"]
    row["A1_A2_A4_padding_mean"] = adaptive_padding / samples
    row["A1_A2_A4_util_pct"] = (
        100.0 * total_assignments / int(adaptive_capacity.sum())
    )
    row["A1_A2_A4_estimated_weight_bytes_mean"] = row[
        "A4_estimated_weight_bytes_mean"
    ]
    row["A1_A2_A4_estimated_weight_bytes_per_useful_assignment"] = row[
        "A4_estimated_weight_bytes_per_useful_assignment"
    ]
    return row


def main() -> None:
    args = parse_args()
    recorder = args.recorder or newest_dump()
    payload = torch.load(recorder, map_location="cpu", weights_only=False)
    counts = payload["logical_count"]
    expected_shape = (args.num_layers, counts.shape[-1])
    if tuple(counts.shape[1:]) != expected_shape:
        raise RuntimeError(
            f"unexpected logical_count shape {tuple(counts.shape)}; "
            f"expected [passes,{args.num_layers},experts]"
        )

    complete_sum = (
        args.batch_size * args.topk * args.num_layers * args.world_size
    )
    complete_mask = counts.sum(dim=(1, 2)) == complete_sum
    complete_indices = torch.nonzero(complete_mask, as_tuple=False).flatten()
    required = args.warmup_passes + args.window_passes
    if complete_indices.numel() < required:
        raise RuntimeError(
            f"only {complete_indices.numel()} complete BS{args.batch_size} passes; "
            f"need {required} ({args.warmup_passes} warm + {args.window_passes} window)"
        )
    selected_indices = complete_indices[
        args.warmup_passes : args.warmup_passes + args.window_passes
    ]
    selected_raw = counts[selected_indices]
    if torch.any(selected_raw.remainder(args.world_size) != 0):
        raise RuntimeError("logical counts are not exactly divisible by world size")
    occupancy = (selected_raw // args.world_size).to(torch.int64)
    expected_assignments = args.batch_size * args.topk * args.num_layers
    if not torch.all(occupancy.sum(dim=(1, 2)) == expected_assignments):
        raise RuntimeError("selected pass assignment checksum mismatch")

    bytes_per_scan = logical_weight_bytes_per_scan(
        args.hidden_size, args.intermediate_per_rank, args.fp4_group_size
    )
    scopes: list[tuple[str, Iterable[int]]] = [
        ("hash", range(0, args.num_hash_layers)),
        ("learned", range(args.num_hash_layers, args.num_layers)),
        ("all", range(0, args.num_layers)),
    ]
    rows: list[dict[str, float | int | str]] = []
    for name, layer_iter in scopes:
        layer_ids = list(layer_iter)
        rows.append(
            summarize(
                occupancy[:, layer_ids, :],
                scope=name,
                passes=args.window_passes,
                layers=len(layer_ids),
                weight_bytes_per_scan=bytes_per_scan,
            )
        )
    if args.include_layers:
        for layer in range(args.num_layers):
            rows.append(
                summarize(
                    occupancy[:, layer : layer + 1, :],
                    scope=f"layer_{layer}",
                    passes=args.window_passes,
                    layers=1,
                    weight_bytes_per_scan=bytes_per_scan,
                )
            )

    metadata = {
        "recorder": str(Path(recorder).resolve()),
        "raw_passes": int(counts.shape[0]),
        "complete_bs32_passes": int(complete_indices.numel()),
        "selected_complete_index_begin": int(selected_indices[0]),
        "selected_complete_index_end": int(selected_indices[-1]),
        "world_size": args.world_size,
        "batch_size": args.batch_size,
        "topk": args.topk,
        "num_layers": args.num_layers,
        "num_hash_layers": args.num_hash_layers,
        "warmup_passes": args.warmup_passes,
        "window_passes": args.window_passes,
        "hidden_size": args.hidden_size,
        "intermediate_per_rank": args.intermediate_per_rank,
        "fp4_group_size": args.fp4_group_size,
        "logical_weight_bytes_per_scan": bytes_per_scan,
        "note": "Estimated bytes are logical kernel weight requests, not measured HBM traffic.",
    }

    fieldnames = list(rows[0].keys())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        stream = args.csv.open("w", newline="")
    else:
        stream = sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.csv:
            stream.close()

    result = {"metadata": metadata, "summaries": rows}
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"analyzed {args.window_passes} passes after {args.warmup_passes} warmup; "
        f"bytes/scan={bytes_per_scan} csv={args.csv or 'stdout'} json={args.json}",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
