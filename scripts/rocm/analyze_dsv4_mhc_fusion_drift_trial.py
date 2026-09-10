#!/usr/bin/env python3
"""Judge the fused-MHC drift trial against a control-vs-control noise floor.

The strict C32 control is not reproducible across rounds, so "the candidate
drifted" proves nothing on its own. The verdict compares cross-arm divergence
(A vs B) with the in-arm floor (A1 vs A2, and each arm's own repeated waves).
A candidate is only implicated when it diverges materially MORE than control
does from itself.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_dsv4_tp8_mhc_fusion_drift_trial import compare  # noqa: E402


def floor_of(arms: dict, names: list[str]) -> list[dict]:
    out = []
    for name in names:
        for cmp in arms[name]["in_arm_comparisons"]:
            out.append(cmp)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    arms = {}
    for name in ("A1", "B1", "B2", "A2"):
        path = args.dir / f"arm_{name}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        arms[name] = json.loads(path.read_text())

    control_waves = arms["A1"]["waves"] + arms["A2"]["waves"]
    cand_waves = arms["B1"]["waves"] + arms["B2"]["waves"]

    # Noise floor: control against itself, across arms and within each arm.
    control_cross = compare(arms["A1"]["waves"][-1], arms["A2"]["waves"][0])
    control_in = floor_of(arms, ["A1", "A2"])
    cand_in = floor_of(arms, ["B1", "B2"])
    cand_cross = compare(arms["B1"]["waves"][-1], arms["B2"]["waves"][0])
    # The comparison of interest.
    a_vs_b = compare(arms["A1"]["waves"][-1], arms["B1"]["waves"][0])

    def rate(cmps: list[dict]) -> float:
        return statistics.fmean([c["differing"] / max(c["requests"], 1) for c in cmps])

    floor = max([control_cross["differing"] / max(control_cross["requests"], 1)]
                + ([rate(control_in)] if control_in else []))
    cand_self = max([cand_cross["differing"] / max(cand_cross["requests"], 1)]
                    + ([rate(cand_in)] if cand_in else []))
    cross = a_vs_b["differing"] / max(a_vs_b["requests"], 1)

    control_tps = [t for name in ("A1", "A2") for t in arms[name]["aggregate_tok_s"]]
    cand_tps = [t for name in ("B1", "B2") for t in arms[name]["aggregate_tok_s"]]
    control_mean = statistics.fmean(control_tps)
    cand_mean = statistics.fmean(cand_tps)
    gain = (cand_mean - control_mean) / control_mean * 100.0

    repeats = sum(len(w["severe_repetition"]) for w in control_waves + cand_waves)
    cand_repeats = sum(len(w["severe_repetition"]) for w in cand_waves)

    print("=== divergence ===")
    print(f"  control self  (floor): {floor:.3f} differing fraction")
    print(f"  candidate self       : {cand_self:.3f}")
    print(f"  control vs candidate : {cross:.3f}")
    print(f"  A-vs-B first_diff_median: {a_vs_b['first_diff_median']}, "
          f"mean shared prefix {a_vs_b['mean_shared_prefix_fraction']:.3f}")
    print("=== throughput ===")
    print(f"  control  : {control_mean:.2f} tok/s {[round(t,2) for t in control_tps]}")
    print(f"  candidate: {cand_mean:.2f} tok/s {[round(t,2) for t in cand_tps]}")
    print(f"  observed : {gain:+.3f}%")
    print("=== quality ===")
    print(f"  severe repetition: candidate {cand_repeats}, total {repeats}")

    # A candidate is implicated only if it exceeds the control's own instability.
    verdict = (
        "candidate_within_control_noise" if cross <= floor + 1e-9
        else "candidate_exceeds_control_noise"
    )
    if cand_repeats:
        verdict = "candidate_quality_regression"
    print(f"VERDICT {verdict}")

    if args.output:
        args.output.write_text(json.dumps({
            "floor_differing_fraction": floor,
            "candidate_self_differing_fraction": cand_self,
            "cross_differing_fraction": cross,
            "a_vs_b": a_vs_b,
            "control_tok_s": control_tps,
            "candidate_tok_s": cand_tps,
            "observed_percent": gain,
            "candidate_severe_repetition": cand_repeats,
            "verdict": verdict,
        }, indent=2, sort_keys=True))
        print(f"WROTE {args.output}")


if __name__ == "__main__":
    main()
