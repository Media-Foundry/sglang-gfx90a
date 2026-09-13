#!/usr/bin/env python3
"""Align one cached row with the same row in a full-recompute eager trace.

This is a diagnostic, not an acceptance threshold. Non-row tensors are
explicitly skipped; logical/cache indices require separate interpretation.
The capture hook synchronizes the observed rank, invalidating timing claims.
"""

import argparse
import json
from pathlib import Path

import torch


def tensor_leaves(value, path="output"):
    if isinstance(value, torch.Tensor):
        yield path, value
    elif isinstance(value, dict):
        for key, child in value.items():
            yield from tensor_leaves(child, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for i, child in enumerate(value):
            yield from tensor_leaves(child, f"{path}.{i}")


def row_metrics(a, b):
    if a.shape != b.shape or a.dtype != b.dtype:
        return {"compatible": False, "shapes": [list(a.shape), list(b.shape)],
                "dtypes": [str(a.dtype), str(b.dtype)]}
    result = {"compatible": True, "shape": list(a.shape), "dtype": str(a.dtype),
              "exact": torch.equal(a, b), "different": int((a != b).sum()),
              "elements": a.numel()}
    if a.is_floating_point():
        aa, bb = a.double().reshape(-1), b.double().reshape(-1)
        result.update(
            finite=bool(torch.isfinite(aa).all() and torch.isfinite(bb).all()),
            max_abs=float((aa - bb).abs().max()) if aa.numel() else 0.0,
            relative_l2=float((aa - bb).norm() / bb.norm().clamp_min(1e-30)),
            cosine=float(torch.nn.functional.cosine_similarity(aa, bb, dim=0)),
            cached_rms=float(aa.square().mean().sqrt()),
            recompute_rms=float(bb.square().mean().sqrt()),
        )
    return result


def compare_records(a, b, cached_rows, full_rows, cached_row, full_row):
    rows = []
    for field in ("args", "output"):
        left = dict(tensor_leaves(a[field], field))
        right = dict(tensor_leaves(b[field], field))
        for path in sorted(left.keys() | right.keys()):
            aa, bb = left.get(path), right.get(path)
            row = {"path": path}
            if aa is None or bb is None:
                row["skipped"] = "missing tensor on one side"
            elif not (aa.ndim and bb.ndim and aa.shape[0] == cached_rows
                      and bb.shape[0] == full_rows):
                row.update(skipped="not a token-row tensor", shapes=[list(aa.shape), list(bb.shape)])
            else:
                row.update(row_metrics(aa[cached_row], bb[full_row]))
            rows.append(row)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--cached-call", type=int, required=True)
    p.add_argument("--full-call", type=int, required=True)
    p.add_argument("--cached-rows", type=int, default=1)
    p.add_argument("--full-rows", type=int, required=True)
    p.add_argument("--cached-row", type=int, default=0)
    p.add_argument("--full-row", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    assert 0 <= args.cached_row < args.cached_rows
    assert 0 <= args.full_row < args.full_rows
    groups = {}
    for file in args.trace_dir.glob("*.pt"):
        if f"-call{args.cached_call}-" not in file.name and f"-call{args.full_call}-" not in file.name:
            continue
        record = torch.load(file, map_location="cpu", weights_only=True)
        if record.get("summary_only"):
            raise ValueError("Full tensor traces required, not sparse row summaries")
        key = (record["rank"], record["module_number"])
        entries = groups.setdefault(key, {})
        assert record["call"] not in entries, "multiple processes/duplicate captures in folder"
        entries[record["call"]] = record
    modules = []
    for (rank, number), entries in sorted(groups.items()):
        a, b = entries.get(args.cached_call), entries.get(args.full_call)
        if a is None or b is None:
            raise ValueError(f"Incomplete module coverage: {(rank, number)}")
        assert (a["prefix"], a["class"]) == (b["prefix"], b["class"])
        modules.append({"rank": rank, "number": number, "prefix": a["prefix"],
                        "class": a["class"], "rows": compare_records(
                            a, b, args.cached_rows, args.full_rows,
                            args.cached_row, args.full_row)})
    assert modules, "empty trace comparison"
    result = {"alignment": vars(args) | {"trace_dir": str(args.trace_dir), "output": str(args.output)},
              "modules": modules}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
    for module in modules:
        outputs = [r for r in module["rows"] if r["path"].startswith("output") and "exact" in r]
        if outputs:
            print(json.dumps({k: module[k] for k in ("rank", "number", "prefix", "class")} |
                             {"outputs": outputs}))


if __name__ == "__main__":
    main()
