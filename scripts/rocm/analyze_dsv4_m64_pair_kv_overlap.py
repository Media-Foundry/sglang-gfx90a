#!/usr/bin/env python3
"""Analyze adjacent DSpark ``[anchor, draft]`` sparse-KV index overlap.

This is an offline, CPU-only feasibility oracle.  It consumes the production
replay payload written by ``dsv4_ck_replay.maybe_dump_unified_sparse`` and does
not import SGLang or initialize HIP.  In particular, compact dump-local indices
are mapped back through ``physical_slots`` before continuity is measured.

Examples::

    python scripts/rocm/analyze_dsv4_m64_pair_kv_overlap.py /tmp/ck-replay
    python scripts/rocm/analyze_dsv4_m64_pair_kv_overlap.py \
        /tmp/ck-replay/*_c128_unified_sparse_m64.pt --json /tmp/pairs.json

The input row contract is the gamma-one target-verify layout
``[anchor_0, draft_0, anchor_1, draft_1, ...]``.  A pair-query kernel is most
promising when set overlap is high *and* the two gather streams contain long
equal runs at a small fixed displacement.  Set overlap alone is only an upper
bound: it does not imply that a lockstep kernel can share loads cheaply.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch


DEFAULT_PATTERN = "*_c128_unified_sparse_m64.pt"


@dataclass(frozen=True)
class PairMetrics:
    dump: str
    pair: int
    anchor_row: int
    draft_row: int
    anchor_position: int | None
    draft_position: int | None
    positions_adjacent: bool | None
    anchor_length: int
    draft_length: int
    intersection: int
    union: int
    jaccard: float
    overlap_min: float
    exact_sequence: bool
    anchor_is_prefix: bool
    draft_is_prefix: bool
    common_prefix: int
    common_prefix_ratio: float
    exact_index_set: bool
    best_shift: int
    best_shift_matches: int
    best_shift_match_ratio: float
    longest_equal_run: int
    shared_tiles: dict[str, int]
    contiguous_physical_tiles: dict[str, int]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    index = fraction * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    weight = index - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _summary(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {key: float("nan") for key in ("mean", "median", "p05", "p95")}
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p05": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
    }


def _common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    length = 0
    for lhs, rhs in zip(a, b):
        if lhs != rhs:
            break
        length += 1
    return length


def _run_lengths(values: Iterable[int]) -> list[int]:
    iterator = iter(values)
    try:
        previous = next(iterator)
    except StopIteration:
        return []
    runs: list[int] = []
    length = 1
    for value in iterator:
        if value == previous + 1:
            length += 1
        else:
            runs.append(length)
            length = 1
        previous = value
    runs.append(length)
    return runs


def _shift_stats(
    a: Sequence[int], b: Sequence[int], shift: int
) -> tuple[int, list[int]]:
    """Return equal elements and equal-run lengths for ``a[i] == b[i+shift]``."""
    a_start = max(0, -shift)
    b_start = max(0, shift)
    count = min(len(a) - a_start, len(b) - b_start)
    if count <= 0:
        return 0, []
    matches = 0
    runs: list[int] = []
    run = 0
    for offset in range(count):
        equal = a[a_start + offset] == b[b_start + offset]
        if equal:
            matches += 1
            run += 1
        elif run:
            runs.append(run)
            run = 0
    if run:
        runs.append(run)
    return matches, runs


def analyze_pair(
    *,
    dump: Path,
    pair: int,
    anchor: Sequence[int],
    draft: Sequence[int],
    anchor_position: int | None,
    draft_position: int | None,
    tile_sizes: Sequence[int],
    max_shift: int,
) -> PairMetrics:
    anchor_set = set(anchor)
    draft_set = set(draft)
    intersection_set = anchor_set & draft_set
    union = len(anchor_set | draft_set)
    intersection = len(intersection_set)
    minimum = min(len(anchor_set), len(draft_set))
    prefix = _common_prefix(anchor, draft)

    shift_candidates = []
    for shift in range(-max_shift, max_shift + 1):
        matches, runs = _shift_stats(anchor, draft, shift)
        # Prefer more reusable entries, then a longer uninterrupted run, then
        # the smallest displacement (and finally the negative shift).
        shift_candidates.append(
            (matches, max(runs, default=0), -abs(shift), -shift, shift, runs)
        )
    _, longest_run, _, _, best_shift, best_runs = max(shift_candidates)
    best_matches, _ = _shift_stats(anchor, draft, best_shift)

    shared_tiles = {
        str(tile): sum(run // tile for run in best_runs) for tile in tile_sizes
    }
    physical_runs = _run_lengths(sorted(intersection_set))
    contiguous_physical_tiles = {
        str(tile): sum(run // tile for run in physical_runs) for tile in tile_sizes
    }

    return PairMetrics(
        dump=str(dump),
        pair=pair,
        anchor_row=2 * pair,
        draft_row=2 * pair + 1,
        anchor_position=anchor_position,
        draft_position=draft_position,
        positions_adjacent=(
            None
            if anchor_position is None or draft_position is None
            else draft_position == anchor_position + 1
        ),
        anchor_length=len(anchor),
        draft_length=len(draft),
        intersection=intersection,
        union=union,
        jaccard=(intersection / union if union else 1.0),
        overlap_min=(intersection / minimum if minimum else 1.0),
        exact_sequence=list(anchor) == list(draft),
        anchor_is_prefix=(
            len(anchor) <= len(draft)
            and list(anchor) == list(draft[: len(anchor)])
        ),
        draft_is_prefix=(
            len(draft) <= len(anchor)
            and list(draft) == list(anchor[: len(draft)])
        ),
        common_prefix=prefix,
        common_prefix_ratio=(prefix / minimum if minimum else 1.0),
        exact_index_set=anchor_set == draft_set,
        best_shift=best_shift,
        best_shift_matches=best_matches,
        best_shift_match_ratio=(best_matches / minimum if minimum else 1.0),
        longest_equal_run=longest_run,
        shared_tiles=shared_tiles,
        contiguous_physical_tiles=contiguous_physical_tiles,
    )


def _as_int_list(tensor: torch.Tensor) -> list[int]:
    return tensor.detach().cpu().to(torch.int64).reshape(-1).tolist()


def analyze_dump(
    path: Path, *, tile_sizes: Sequence[int], max_shift: int
) -> tuple[list[PairMetrics], dict]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise TypeError(f"{path}: expected a dict payload, got {type(payload).__name__}")
    if "kv_indices" not in payload or "kv_indptr" not in payload:
        raise KeyError(f"{path}: missing kv_indices/kv_indptr")

    compact_indices = payload["kv_indices"].to(torch.int64).reshape(-1)
    indptr = _as_int_list(payload["kv_indptr"])
    if len(indptr) < 2 or indptr[0] != 0:
        raise ValueError(f"{path}: kv_indptr must start at zero and contain rows")
    if any(lhs > rhs for lhs, rhs in zip(indptr, indptr[1:])):
        raise ValueError(f"{path}: kv_indptr is not monotonic")
    if indptr[-1] > compact_indices.numel():
        raise ValueError(
            f"{path}: terminal offset {indptr[-1]} exceeds {compact_indices.numel()} indices"
        )
    rows = len(indptr) - 1
    if rows % 2:
        raise ValueError(
            f"{path}: adjacent pair layout requires an even row count, got {rows}"
        )

    physical_slots = payload.get("physical_slots")
    if physical_slots is not None:
        physical_slots = physical_slots.to(torch.int64).reshape(-1)
        valid = compact_indices[: indptr[-1]]
        if valid.numel() and (
            int(valid.min()) < 0 or int(valid.max()) >= physical_slots.numel()
        ):
            raise ValueError(f"{path}: compact index is outside physical_slots")
        indices = physical_slots.index_select(0, valid)
        namespace = "physical_slots"
    else:
        indices = compact_indices[: indptr[-1]]
        namespace = "payload_indices_without_physical_map"

    positions_tensor = payload.get("positions")
    positions = _as_int_list(positions_tensor) if positions_tensor is not None else None
    if positions is not None and len(positions) != rows:
        raise ValueError(f"{path}: positions has {len(positions)} rows, expected {rows}")

    pairs: list[PairMetrics] = []
    indices_list = _as_int_list(indices)
    for pair in range(rows // 2):
        a_row, d_row = 2 * pair, 2 * pair + 1
        anchor = indices_list[indptr[a_row] : indptr[a_row + 1]]
        draft = indices_list[indptr[d_row] : indptr[d_row + 1]]
        pairs.append(
            analyze_pair(
                dump=path,
                pair=pair,
                anchor=anchor,
                draft=draft,
                anchor_position=(positions[a_row] if positions is not None else None),
                draft_position=(positions[d_row] if positions is not None else None),
                tile_sizes=tile_sizes,
                max_shift=max_shift,
            )
        )

    provenance = payload.get("provenance", {})
    metadata = {
        "path": str(path),
        "rows": rows,
        "pairs": rows // 2,
        "index_namespace": namespace,
        "provenance": provenance,
    }
    return pairs, metadata


def aggregate(pairs: Sequence[PairMetrics], tile_sizes: Sequence[int]) -> dict:
    if not pairs:
        raise ValueError("no anchor/draft pairs were analyzed")
    total_indices = sum(item.anchor_length + item.draft_length for item in pairs)
    total_intersection = sum(item.intersection for item in pairs)
    total_best_matches = sum(item.best_shift_matches for item in pairs)
    adjacent = [
        item.positions_adjacent
        for item in pairs
        if item.positions_adjacent is not None
    ]
    return {
        "pairs": len(pairs),
        "jaccard": _summary([item.jaccard for item in pairs]),
        "overlap_min": _summary([item.overlap_min for item in pairs]),
        "common_prefix_ratio": _summary([item.common_prefix_ratio for item in pairs]),
        "best_shift_match_ratio": _summary(
            [item.best_shift_match_ratio for item in pairs]
        ),
        "position_adjacent_fraction": (
            sum(bool(value) for value in adjacent) / len(adjacent) if adjacent else None
        ),
        "exact_sequence_fraction": sum(item.exact_sequence for item in pairs)
        / len(pairs),
        "exact_index_set_fraction": sum(item.exact_index_set for item in pairs)
        / len(pairs),
        "anchor_prefix_fraction": sum(item.anchor_is_prefix for item in pairs)
        / len(pairs),
        "either_prefix_fraction": sum(
            item.anchor_is_prefix or item.draft_is_prefix for item in pairs
        )
        / len(pairs),
        # If every shared slot could be loaded once instead of twice.  This is
        # an optimistic set-level byte bound, not a predicted kernel speedup.
        "set_reuse_load_reduction_upper_bound": (
            total_intersection / total_indices if total_indices else 0.0
        ),
        # Entries reusable by a lockstep stream after selecting one small
        # displacement per pair.  Still optimistic: register/CTA costs remain.
        "best_shift_load_reduction_upper_bound": (
            total_best_matches / total_indices if total_indices else 0.0
        ),
        "best_shift_histogram": dict(
            sorted(Counter(str(item.best_shift) for item in pairs).items())
        ),
        "longest_equal_run": _summary(
            [float(item.longest_equal_run) for item in pairs]
        ),
        "shared_tiles": {
            str(tile): sum(item.shared_tiles[str(tile)] for item in pairs)
            for tile in tile_sizes
        },
        "contiguous_physical_tiles": {
            str(tile): sum(
                item.contiguous_physical_tiles[str(tile)] for item in pairs
            )
            for tile in tile_sizes
        },
    }


def resolve_inputs(values: Sequence[str], pattern: str) -> list[Path]:
    paths: set[Path] = set()
    for value in values:
        path = Path(value)
        if path.is_dir():
            paths.update(item for item in path.rglob(pattern) if item.is_file())
        elif path.is_file():
            paths.add(path)
        else:
            # Path.glob does not accept absolute patterns, so split at the
            # final slash and glob from its parent for simple shell-free use.
            parent = path.parent if str(path.parent) else Path(".")
            paths.update(item for item in parent.glob(path.name) if item.is_file())
    return sorted(item.resolve() for item in paths)


def _parse_tile_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(sorted({int(item) for item in value.split(",") if item.strip()}))
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError(
            "tile sizes must be positive comma-separated integers"
        )
    return sizes


def _print_summary(summary: dict) -> None:
    def metric(name: str) -> str:
        item = summary[name]
        return (
            f"mean={item['mean']:.6f} median={item['median']:.6f} "
            f"p05={item['p05']:.6f} p95={item['p95']:.6f}"
        )

    print(f"pairs={summary['pairs']}")
    print(f"jaccard {metric('jaccard')}")
    print(f"overlap_min {metric('overlap_min')}")
    print(f"common_prefix_ratio {metric('common_prefix_ratio')}")
    print(f"best_shift_match_ratio {metric('best_shift_match_ratio')}")
    print(f"position_adjacent_fraction={summary['position_adjacent_fraction']}")
    print(f"exact_sequence_fraction={summary['exact_sequence_fraction']:.6f}")
    print(f"exact_index_set_fraction={summary['exact_index_set_fraction']:.6f}")
    print(f"anchor_prefix_fraction={summary['anchor_prefix_fraction']:.6f}")
    print(f"either_prefix_fraction={summary['either_prefix_fraction']:.6f}")
    print(
        "set_reuse_load_reduction_upper_bound="
        f"{summary['set_reuse_load_reduction_upper_bound']:.6f}"
    )
    print(
        "best_shift_load_reduction_upper_bound="
        f"{summary['best_shift_load_reduction_upper_bound']:.6f}"
    )
    print(f"best_shift_histogram={summary['best_shift_histogram']}")
    print(f"longest_equal_run {metric('longest_equal_run')}")
    print(f"shared_tiles={summary['shared_tiles']}")
    print(f"contiguous_physical_tiles={summary['contiguous_physical_tiles']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="replay .pt files, directories, or globs"
    )
    parser.add_argument(
        "--pattern", default=DEFAULT_PATTERN, help="recursive directory pattern"
    )
    parser.add_argument("--tile-sizes", type=_parse_tile_sizes, default=(8, 16, 32))
    parser.add_argument("--max-shift", type=int, default=32)
    parser.add_argument("--expected-rows", type=int, default=64)
    parser.add_argument("--expected-compress-ratio", type=int, default=128)
    parser.add_argument("--json", type=Path, help="write aggregate and per-pair JSON")
    args = parser.parse_args()
    if args.max_shift < 0:
        parser.error("--max-shift must be non-negative")

    paths = resolve_inputs(args.inputs, args.pattern)
    if not paths:
        parser.error(
            f"no replay dumps found (directory pattern: {args.pattern!r}); "
            "capture with SGLANG_DSV4_CK_REPLAY_KINDS=unified_sparse, "
            "SGLANG_DSV4_CK_REPLAY_ROWS=64, and decode graphs disabled"
        )

    all_pairs: list[PairMetrics] = []
    dumps: list[dict] = []
    for path in paths:
        pairs, metadata = analyze_dump(
            path, tile_sizes=args.tile_sizes, max_shift=args.max_shift
        )
        if args.expected_rows > 0 and metadata["rows"] != args.expected_rows:
            raise ValueError(
                f"{path}: expected {args.expected_rows} M64 rows, "
                f"got {metadata['rows']}"
            )
        compress_ratio = metadata["provenance"].get("compress_ratio")
        if (
            args.expected_compress_ratio > 0
            and compress_ratio != args.expected_compress_ratio
        ):
            raise ValueError(
                f"{path}: expected C{args.expected_compress_ratio}, "
                f"got provenance compress_ratio={compress_ratio!r}"
            )
        all_pairs.extend(pairs)
        dumps.append(metadata)
        print(
            f"DUMP path={path} pairs={len(pairs)} "
            f"compress_ratio={compress_ratio} "
            f"index_namespace={metadata['index_namespace']}"
        )

    summary = aggregate(all_pairs, args.tile_sizes)
    _print_summary(summary)
    if args.json is not None:
        output = {
            "schema": "dsv4_dspark_gamma1_pair_kv_overlap_v1",
            "tile_sizes": list(args.tile_sizes),
            "max_shift": args.max_shift,
            "dumps": dumps,
            "summary": summary,
            "pairs": [asdict(item) for item in all_pairs],
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n")
        print(f"json={args.json}")


if __name__ == "__main__":
    main()
