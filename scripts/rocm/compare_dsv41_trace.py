#!/usr/bin/env python3
"""Compare full-byte output hashes from bounded repeated-prefill hook traces.

Sample errors are NOT full-tensor norms. Unequal hashes prove a difference;
matching sampled rows alone do not prove equality. Default compares two-chunk
requests, with the first request's two calls as reference for later requests.
"""

import argparse
import gzip
import json
from pathlib import Path

import torch


def tensors(value, path="output"):
    if isinstance(value, dict):
        if "sha256" in value and "sample" in value:
            yield path, value
        else:
            for name, child in value.items():
                yield from tensors(child, f"{path}.{name}")
    elif isinstance(value, (tuple, list)):
        for i, child in enumerate(value):
            yield from tensors(child, f"{path}[{i}]")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calls-per-request", type=int, default=2)
    parser.add_argument("--requests", type=int, default=3)
    args = parser.parse_args()
    assert args.calls_per_request > 0 and args.requests > 1
    records = {}
    for path in sorted(args.trace_dir.glob("*.pt")):
        value = torch.load(path, weights_only=True, map_location="cpu")
        assert value["summary_only"], path
        key = value["rank"], value["module_number"]
        assert value["call"] not in records.setdefault(key, {}), path
        records[key][value["call"]] = value
    assert records, "No trace records"
    comparisons, missing = [], []
    total_calls = args.requests * args.calls_per_request
    for (rank, number), calls in sorted(records.items()):
        if set(calls) != set(range(total_calls)):
            missing.append({"rank": rank, "module_number": number, "calls": sorted(calls)})
            continue
        for call in range(args.calls_per_request, total_calls):
            base = calls[call % args.calls_per_request]
            new = calls[call]
            aa, bb = dict(tensors(base["output"])), dict(tensors(new["output"]))
            # Logical selection is comparable across fresh allocations;
            # physical page IDs need not be, and are deliberately excluded.
            for record, target in ((base, aa), (new, bb)):
                logical = record.get("sparse_metadata", {}).get("logical")
                target.update(tensors(logical, "logical_indices"))
            assert aa and aa.keys() == bb.keys(), (rank, number, call)
            for name in aa:
                a, b = aa[name], bb[name]
                row = {"rank": rank, "module_number": number, "class": new["class"],
                       "layer_id": new["layer_id"], "call": call, "tensor": name,
                       "shape": a["shape"], "shape_equal": a["shape"] == b["shape"],
                       "reference_sha256": a["sha256"], "actual_sha256": b["sha256"],
                       "full_hash_equal": a["sha256"] == b["sha256"]}
                if row["shape_equal"]:
                    assert a["sample_rows"] == b["sample_rows"]
                    sa, sb = a["sample"].float(), b["sample"].float()
                    diff = sb - sa
                    row.update(sample_exact=torch.equal(a["sample"], b["sample"]),
                        sample_max_abs=float(diff.abs().max()) if diff.numel() else 0.0,
                        sample_relative_l2=float(diff.norm() / sa.norm().clamp_min(1e-30)),
                        sample_finite=bool(torch.isfinite(sb).all()))
                comparisons.append(row)
    report = {"trace_dir": str(args.trace_dir), "modules": len(records),
              "coverage_complete": not missing, "missing_calls": missing,
              "comparisons": comparisons}
    opener = gzip.open if args.output.suffix == ".gz" else open
    with opener(args.output, "xt") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps({"modules": len(records), "missing": missing,
        "first_differences": {
            str(call): [r for r in comparisons if r["call"] == call and not r["full_hash_equal"]][:12]
            for call in range(args.calls_per_request, total_calls)
        }}, indent=2), flush=True)
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
