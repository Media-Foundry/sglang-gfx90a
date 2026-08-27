#!/usr/bin/env python3
"""Held-out hot-expert cache analysis for TP8 BS32 DSV4 recorder dumps.

After selecting complete BS32 decode passes and dropping the warm prefix, the
first half of the requested window chooses per-layer hot experts.  The second
half reports assignment and A4 weight-scan hit rates.  Logical w13/w2 bytes are
derived from the current TP8 H4096/I256 packed-FP4 shapes; they are not hardware
counter measurements.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import torch


TOP_NS = (8, 16, 32, 64)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "recorder",
        nargs="?",
        help="expert_distribution_recorder_*.pt (defaults to newest by mtime)",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--world-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--topk", type=int, default=6)
    parser.add_argument("--num-layers", type=int, default=43)
    parser.add_argument("--num-hash-layers", type=int, default=3)
    parser.add_argument("--warmup-passes", type=int, default=32)
    parser.add_argument("--train-passes", type=int, default=64)
    parser.add_argument("--test-passes", type=int, default=64)
    parser.add_argument("--assignment-tile", type=int, default=4)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-per-rank", type=int, default=256)
    parser.add_argument("--fp4-group-size", type=int, default=32)
    parser.add_argument("--continue-scan-hit-pct", type=float, default=46.0)
    return parser.parse_args()


def newest_dump() -> str:
    candidates = glob.glob("/tmp/expert_distribution_recorder_*.pt")
    if not candidates:
        raise FileNotFoundError("no /tmp expert distribution recorder dump found")
    return max(candidates, key=lambda path: Path(path).stat().st_mtime_ns)


def packed_bytes_per_scan(hidden: int, intermediate: int, group: int) -> tuple[int, int]:
    if hidden % group or intermediate % group:
        raise ValueError("hidden and intermediate must be divisible by fp4 group size")
    # w13 [2I,H]: packed FP4 plus E8M0 byte per group-32.
    w13 = (2 * intermediate * hidden) // 2 + 2 * intermediate * (hidden // group)
    # w2 [H,I]: packed FP4 plus E8M0 byte per group-32.
    w2 = (hidden * intermediate) // 2 + hidden * (intermediate // group)
    return w13, w2


def pct(numerator: int, denominator: int) -> float:
    return 100.0 * numerator / denominator if denominator else 0.0


def quantiles(values: list[float]) -> dict[str, float]:
    tensor = torch.tensor(values, dtype=torch.float64)
    return {
        "p50": torch.quantile(tensor, 0.50).item(),
        "p95": torch.quantile(tensor, 0.95).item(),
    }


def group_summary(layers: list[dict], name: str) -> dict:
    assign_hit = sum(layer["assignment_hit"] for layer in layers)
    assign_total = sum(layer["assignment_total"] for layer in layers)
    scan_hit = sum(layer["a4_scan_hit"] for layer in layers)
    scan_total = sum(layer["a4_scan_total"] for layer in layers)
    return {
        "scope": name,
        "layers": len(layers),
        "assignment_hit": assign_hit,
        "assignment_total": assign_total,
        "assignment_hit_pct": pct(assign_hit, assign_total),
        "assignment_hit_pct_per_layer": quantiles(
            [layer["assignment_hit_pct"] for layer in layers]
        ),
        "a4_scan_hit": scan_hit,
        "a4_scan_total": scan_total,
        "a4_scan_hit_pct": pct(scan_hit, scan_total),
        "a4_scan_hit_pct_per_layer": quantiles(
            [layer["a4_scan_hit_pct"] for layer in layers]
        ),
        "w13_logical_byte_hit_pct": pct(scan_hit, scan_total),
        "w2_logical_byte_hit_pct": pct(scan_hit, scan_total),
    }


def main() -> None:
    args = parse_args()
    recorder = args.recorder or newest_dump()
    payload = torch.load(recorder, map_location="cpu", weights_only=False)
    counts = payload["logical_count"]
    if counts.ndim != 3 or counts.shape[1] != args.num_layers:
        raise RuntimeError(f"unexpected logical_count shape: {tuple(counts.shape)}")
    complete_sum = args.batch_size * args.topk * args.num_layers * args.world_size
    complete_indices = torch.nonzero(
        counts.sum(dim=(1, 2)) == complete_sum, as_tuple=False
    ).flatten()
    required = args.warmup_passes + args.train_passes + args.test_passes
    if complete_indices.numel() < required:
        raise RuntimeError(
            f"only {complete_indices.numel()} complete passes; need {required}"
        )
    selected_indices = complete_indices[args.warmup_passes : required]
    selected_raw = counts[selected_indices]
    if torch.any(selected_raw.remainder(args.world_size) != 0):
        raise RuntimeError("logical counts are not divisible by world size")
    occupancy = (selected_raw // args.world_size).to(torch.int64)
    expected = args.batch_size * args.topk * args.num_layers
    if not torch.all(occupancy.sum(dim=(1, 2)) == expected):
        raise RuntimeError("selected pass assignment checksum mismatch")

    train = occupancy[: args.train_passes]
    test = occupancy[args.train_passes :]
    train_counts = train.sum(dim=0)
    # Stable sorting makes equal-count ties reproducible by expert ID.
    hot_order = torch.argsort(train_counts, dim=-1, descending=True, stable=True)
    tile = args.assignment_tile
    test_blocks = torch.div(test + tile - 1, tile, rounding_mode="floor")
    w13_bytes, w2_bytes = packed_bytes_per_scan(
        args.hidden_size, args.intermediate_per_rank, args.fp4_group_size
    )

    analyses: dict[str, dict] = {}
    for top_n in TOP_NS:
        layer_rows: list[dict] = []
        for layer in range(args.num_layers):
            hot = hot_order[layer, :top_n]
            layer_test = test[:, layer, :]
            layer_blocks = test_blocks[:, layer, :]
            assignment_hit = int(layer_test[:, hot].sum())
            assignment_total = int(layer_test.sum())
            scan_hit = int(layer_blocks[:, hot].sum())
            scan_total = int(layer_blocks.sum())
            layer_rows.append(
                {
                    "layer": layer,
                    "router": "hash" if layer < args.num_hash_layers else "learned",
                    "hot_expert_ids": hot.tolist(),
                    "assignment_hit": assignment_hit,
                    "assignment_total": assignment_total,
                    "assignment_hit_pct": pct(assignment_hit, assignment_total),
                    "a4_scan_hit": scan_hit,
                    "a4_scan_total": scan_total,
                    "a4_scan_hit_pct": pct(scan_hit, scan_total),
                    "w13_logical_byte_hit": scan_hit * w13_bytes,
                    "w13_logical_byte_total": scan_total * w13_bytes,
                    "w13_logical_byte_hit_pct": pct(scan_hit, scan_total),
                    "w2_logical_byte_hit": scan_hit * w2_bytes,
                    "w2_logical_byte_total": scan_total * w2_bytes,
                    "w2_logical_byte_hit_pct": pct(scan_hit, scan_total),
                }
            )
        hash_rows = layer_rows[: args.num_hash_layers]
        learned_rows = layer_rows[args.num_hash_layers :]
        all_summary = group_summary(layer_rows, "all")
        hash_summary = group_summary(hash_rows, "hash")
        learned_summary = group_summary(learned_rows, "learned")
        for summary in (all_summary, hash_summary, learned_summary):
            summary["w13_logical_byte_hit"] = summary["a4_scan_hit"] * w13_bytes
            summary["w13_logical_byte_total"] = summary["a4_scan_total"] * w13_bytes
            summary["w2_logical_byte_hit"] = summary["a4_scan_hit"] * w2_bytes
            summary["w2_logical_byte_total"] = summary["a4_scan_total"] * w2_bytes

        # A cached signed-INT8 w2 codebook needs H*I bytes per expert/layer;
        # the existing E8M0 scales are reused and are not duplicated.
        w2_cache_bytes_per_expert_layer = args.hidden_size * args.intermediate_per_rank
        cache_bytes_hash = top_n * args.num_hash_layers * w2_cache_bytes_per_expert_layer
        cache_bytes_learned = (
            top_n
            * (args.num_layers - args.num_hash_layers)
            * w2_cache_bytes_per_expert_layer
        )
        cache_bytes_all = cache_bytes_hash + cache_bytes_learned
        analyses[str(top_n)] = {
            "top_n": top_n,
            "summaries": {
                "hash": hash_summary,
                "learned": learned_summary,
                "all": all_summary,
            },
            "w2_only_int8_cache": {
                "bytes_per_expert_layer": w2_cache_bytes_per_expert_layer,
                "hash_layers_bytes": cache_bytes_hash,
                "learned_layers_bytes": cache_bytes_learned,
                "all_layers_bytes": cache_bytes_all,
                "hash_layers_gib": cache_bytes_hash / 2**30,
                "learned_layers_gib": cache_bytes_learned / 2**30,
                "all_layers_gib": cache_bytes_all / 2**30,
                "reuses_existing_weight_scales": True,
            },
            "continue_gate": {
                "threshold_a4_scan_hit_pct": args.continue_scan_hit_pct,
                "learned_passes": (
                    learned_summary["a4_scan_hit_pct"]
                    >= args.continue_scan_hit_pct
                ),
                "all_passes": (
                    all_summary["a4_scan_hit_pct"] >= args.continue_scan_hit_pct
                ),
            },
            "per_layer": layer_rows,
        }

    result = {
        "format": "dsv4-tp8-bs32-heldout-hot-expert-cache-v1",
        "metadata": {
            "recorder": str(Path(recorder).resolve()),
            "raw_shape": list(counts.shape),
            "complete_passes": int(complete_indices.numel()),
            "selected_raw_indices": selected_indices.tolist(),
            "warmup_complete_passes": args.warmup_passes,
            "train_passes": args.train_passes,
            "test_passes": args.test_passes,
            "world_size": args.world_size,
            "batch_size": args.batch_size,
            "topk": args.topk,
            "num_layers": args.num_layers,
            "num_hash_layers": args.num_hash_layers,
            "assignment_tile": tile,
            "hidden_size": args.hidden_size,
            "intermediate_per_rank": args.intermediate_per_rank,
            "w13_logical_bytes_per_a4_scan": w13_bytes,
            "w2_logical_bytes_per_a4_scan": w2_bytes,
            "logical_bytes_note": "Kernel-requested bytes, not measured HBM traffic.",
        },
        "top_n": analyses,
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
