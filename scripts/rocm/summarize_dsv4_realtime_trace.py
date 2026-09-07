#!/usr/bin/env python3
"""Summarize matched TP-rank timestamp samples, not summed multi-stream work.

Requires replay IDs from the nonblocking logger. Invalid fine-grained markers
do not invalidate an independently complete/monotonic coarse layer interval.
The historical gfx90a s_memrealtime calibration is explicitly configurable.
"""

import argparse
import json
import re
import statistics
from pathlib import Path


COARSE = (
    "attn_mhc_norm", "attention_entry", "attention_prepare", "attention_core",
    "attention_output_collective", "ffn_mhc_norm", "moe_collective",
)


def summarize(text, ranks=8, tick_us=0.04):
    records = {}
    for line in text.splitlines():
        if "realtime layer trace:" not in line:
            continue
        m = re.search(r"rank=(\d+) ticks=(\[[^\]]+\]).*replay=(\d+)", line)
        if not m:
            continue
        rank, ticks, replay = int(m[1]), json.loads(m[2]), int(m[3])
        if len(ticks) < 8 or any(t <= 0 for t in ticks[:8]):
            continue
        if any(ticks[j + 1] < ticks[j] for j in range(7)):
            continue
        key = (replay, rank)
        if key in records:
            raise ValueError(f"Duplicate trace sample {key}; use one server log")
        records[key] = [
            (ticks[j + 1] - ticks[j]) * tick_us for j in range(7)
        ] + [(ticks[7] - ticks[0]) * tick_us]
    matched = sorted(
        replay for replay in {key[0] for key in records}
        if all((replay, rank) in records for rank in range(ranks))
    )
    if not matched:
        raise ValueError("No complete, monotonic, replay-ID-matched rank samples")
    names = COARSE + ("whole_layer",)
    result = {
        "tick_us": tick_us, "ranks": ranks,
        "valid_rank_samples": len(records), "matched_replays": matched,
        "rankmax_median_us": {}, "per_rank_median_us": {},
        "warning": "Instrumented spans include marker and arrival-wait costs; "
                   "the sum of component rank-max medians is not a critical path.",
    }
    for j, name in enumerate(names):
        result["rankmax_median_us"][name] = statistics.median(
            max(records[replay, rank][j] for rank in range(ranks))
            for replay in matched
        )
        result["per_rank_median_us"][name] = [
            statistics.median(records[replay, rank][j] for replay in matched)
            for rank in range(ranks)
        ]
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("log", type=Path)
    p.add_argument("--ranks", type=int, default=8)
    p.add_argument("--tick-us", type=float, default=0.04)
    a = p.parse_args()
    print(json.dumps(summarize(a.log.read_text(), a.ranks, a.tick_us), indent=2))
