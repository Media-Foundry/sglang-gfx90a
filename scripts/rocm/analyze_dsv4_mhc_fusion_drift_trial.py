#!/usr/bin/env python3
"""Judge the fused-MHC drift trial against a control-vs-control noise floor.

Arm A1 measured a 32/32 in-arm divergence with no candidate change, so a
differing-request *count* saturates at 1.0 and cannot discriminate. This reads
metrics that still have resolution at saturation -- where divergence starts and
how much prefix survives -- and reports the fp16-weight and reduction-order
effects separately via the C arms.

Nothing here decides on divergence alone: a candidate is only implicated when it
falls outside the spread control shows against itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_dsv4_tp8_mhc_fusion_drift_trial import compare  # noqa: E402

FAMILIES = {"A": "control", "B": "candidate_fp16_weights", "C": "candidate_fp32_weights"}


def load(directory: Path) -> dict:
    arms = {}
    for path in sorted(directory.glob("arm_*.json")):
        arms[path.stem.replace("arm_", "")] = json.loads(path.read_text())
    if not arms:
        raise SystemExit(f"no arm_*.json under {directory}")
    return arms


def family_of(arm: str) -> str:
    return arm[0]


def prefix_stats(comparisons: list[dict]) -> dict:
    """Divergence-onset metrics, which stay informative when counts saturate."""
    prefixes = [c["mean_shared_prefix_fraction"] for c in comparisons]
    firsts = [c["first_diff_median"] for c in comparisons
              if c["first_diff_median"] is not None]
    return {
        "n": len(comparisons),
        "mean_shared_prefix": statistics.fmean(prefixes) if prefixes else None,
        "shared_prefix_spread": (max(prefixes) - min(prefixes)) if len(prefixes) > 1 else 0.0,
        "median_first_diff": statistics.median(firsts) if firsts else None,
        "differing_fraction": statistics.fmean(
            [c["differing"] / max(c["requests"], 1) for c in comparisons]
        ) if comparisons else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arms = load(args.dir)
    print(f"arms present: {sorted(arms)}")

    # Every within-family comparison is a same-config pair, so it measures only
    # run-to-run instability -- the floor a candidate must be judged against.
    within: dict[str, list[dict]] = {key: [] for key in FAMILIES}
    for name, arm in arms.items():
        within[family_of(name)].extend(arm["in_arm_comparisons"])
    for family in FAMILIES:
        names = sorted(n for n in arms if family_of(n) == family)
        for i in range(len(names) - 1):
            left, right = arms[names[i]], arms[names[i + 1]]
            within[family].append(compare(left["waves"][-1], right["waves"][0]))

    print("\n=== within-family (same config; this is the floor) ===")
    floors = {}
    for family, label in FAMILIES.items():
        if not within[family]:
            continue
        stats = prefix_stats(within[family])
        floors[family] = stats
        print(f"  {label:24s} n={stats['n']} "
              f"differing={stats['differing_fraction']:.3f} "
              f"shared_prefix={stats['mean_shared_prefix']:.4f} "
              f"(spread {stats['shared_prefix_spread']:.4f}) "
              f"first_diff={stats['median_first_diff']}")

    print("\n=== cross-family ===")
    cross = {}
    for a, b in (("A", "B"), ("A", "C"), ("B", "C")):
        pairs = []
        for left in sorted(n for n in arms if family_of(n) == a):
            for right in sorted(n for n in arms if family_of(n) == b):
                pairs.append(compare(arms[left]["waves"][-1], arms[right]["waves"][0]))
        if not pairs:
            continue
        stats = prefix_stats(pairs)
        cross[f"{a}v{b}"] = stats
        print(f"  {a} vs {b}: n={stats['n']} "
              f"differing={stats['differing_fraction']:.3f} "
              f"shared_prefix={stats['mean_shared_prefix']:.4f} "
              f"first_diff={stats['median_first_diff']}")

    print("\n=== throughput ===")
    tps, accepts = {}, {}
    for family, label in FAMILIES.items():
        # Resident window only: whole-wave rates include the admission ramp,
        # whose 0 -> 32 batch growth dwarfs the effect under test.
        values = [w["resident"]["tok_s"] for n in arms if family_of(n) == family
                  for w in arms[n]["waves"]
                  if w.get("resident", {}).get("tok_s") is not None]
        acc = [a for n in arms if family_of(n) == family
               for w in arms[n]["waves"] for a in w.get("accept_lengths", [])]
        acc = [statistics.fmean(acc)] if acc else []
        if not values:
            continue
        tps[family] = values
        accepts[family] = acc
        spread = (max(values) - min(values)) / statistics.fmean(values) * 100.0
        print(f"  {label:24s} mean={statistics.fmean(values):8.2f} tok/s "
              f"spread={spread:5.1f}% {[round(v, 1) for v in values]}")
        print(f"  {'':24s} accept_len={[round(a, 3) for a in acc] or 'unavailable'}")

    print("\n=== quality ===")
    for family, label in FAMILIES.items():
        flagged = [r for n in arms if family_of(n) == family
                   for w in arms[n]["waves"] for r in w["severe_repetition"]]
        if any(family_of(n) == family for n in arms):
            print(f"  {label:24s} severe_repetition={len(flagged)} {sorted(set(flagged))}")

    # Verdict. Control's own instability bounds what any candidate can be blamed
    # for; a throughput spread wider than the effect under test means the arms
    # cannot resolve it regardless of how the divergence reads.
    notes = []
    if "A" in floors and floors["A"]["differing_fraction"] is not None:
        if floors["A"]["differing_fraction"] > 0.5:
            notes.append(
                f"control diverges from itself on "
                f"{floors['A']['differing_fraction']:.0%} of requests; "
                "differing-count cannot discriminate"
            )
    if "A" in tps and len(tps["A"]) > 1:
        spread = (max(tps["A"]) - min(tps["A"])) / statistics.fmean(tps["A"]) * 100.0
        if spread > 5.0:
            notes.append(
                f"control throughput spread {spread:.1f}% exceeds the ~2.4% the "
                "whole MHC pool can deliver; this trial cannot resolve the gain"
            )
    verdict = "inconclusive_control_unstable" if notes else "measurable"
    print(f"\nVERDICT {verdict}")
    for note in notes:
        print(f"  - {note}")

    if args.output:
        args.output.write_text(json.dumps({
            "arms": sorted(arms),
            "within_family": floors,
            "cross_family": cross,
            "throughput": tps,
            "accept_lengths": accepts,
            "verdict": verdict,
            "notes": notes,
        }, indent=2, sort_keys=True))
        print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
