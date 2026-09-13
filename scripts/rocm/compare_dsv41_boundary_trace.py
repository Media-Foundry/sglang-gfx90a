"""Compare fixed-prefix HC/Engram traces, with explicit per-method call maps."""

import argparse
import json
from pathlib import Path

import torch

from compare_dsv41_row_trace import compare_records


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cached-forward", type=int, default=1)
    p.add_argument("--full-forward", type=int, default=3)
    p.add_argument("--full-rows", type=int, default=204)
    args = p.parse_args()
    groups = {}
    for file in sorted(args.trace_dir.glob("*.pt")):
        record = torch.load(file, weights_only=True, map_location="cpu")
        calls_per_forward = 1 if record["tag"] == "engram_gate" else 2
        forward, sublayer = divmod(record["call"], calls_per_forward)
        if forward not in (args.cached_forward, args.full_forward):
            continue
        module = file.name.split("-module")[1].split("-")[0]
        key = (record["rank"], record["tag"], module, sublayer)
        entries = groups.setdefault(key, {})
        assert forward not in entries, f"duplicate record {file}"
        entries[forward] = record
    assert groups, "empty boundary capture"
    results = []
    for key, entries in sorted(groups.items()):
        a, b = entries[args.cached_forward], entries[args.full_forward]
        rows = compare_records(a, b, 1, args.full_rows, 0, args.full_rows - 1)
        assert any("exact" in r for r in rows), "no compared tensor rows"
        result = {"key": key, "layer_id": a["layer_id"], "prefix": a["prefix"], "rows": rows}
        results.append(result)
        differences = [{k: r[k] for k in ("path", "different", "max_abs") if k in r}
                       for r in rows if r.get("exact") is False]
        print(json.dumps({"key": key, "layer": a["layer_id"], "prefix": a["prefix"],
                          "differences": differences}))
    with args.output.open("x") as f:
        json.dump({"trace_dir": str(args.trace_dir), "cached_forward": args.cached_forward,
                   "full_forward": args.full_forward, "full_rows": args.full_rows,
                   "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
