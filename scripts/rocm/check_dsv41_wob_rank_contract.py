#!/usr/bin/env python3
"""Replay each captured rank-local wo_b, then inspect the TP reduction.

Uses actual runtime FP8 weights/scales and hidden inputs, not synthetic or
constant-scale data. This process does not modify or communicate with service
weights. Run on an idle GCD with sufficient free memory; no timing claim.
"""

import argparse
import json
from pathlib import Path

import torch

from scripts.rocm.compare_dsv41_row_trace import row_metrics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from sglang.srt.layers.quantization.fp8_utils import dispatch_w8a8_block_fp8_linear

    linear = dispatch_w8a8_block_fp8_linear([32, 32], act_scale_ue8m0=True)
    groups = {}
    for file in args.trace_dir.glob("*RowParallelLinear*-call1-*.pt"):
        a = torch.load(file, map_location="cpu", weights_only=True)
        b = torch.load(str(file).replace("-call1-", "-call3-"), map_location="cpu", weights_only=True)
        groups.setdefault(a["module_number"], []).append((a, b))
    results = []
    for module_number, records in sorted(groups.items()):
        assert len(records) == 8, "need all eight rank partials"
        per_rank, left, right = [], [], []
        for a, b in sorted(records, key=lambda pair: pair[0]["rank"]):
            params = a["parameters"]
            w = params["weight"].cuda()
            s = params["weight_scale_inv"].cuda()
            x = a["args"][0].cuda()
            full_x = b["args"][0].cuda()
            ya = linear(x, w, [32, 32], s).cpu()
            yb = linear(full_x, w, [32, 32], s).cpu()[-1:]
            left.append(ya)
            right.append(yb)
            per_rank.append({"rank": a["rank"], "input": row_metrics(x.cpu(), full_x[-1:].cpu()),
                             "local_output": row_metrics(ya, yb)})
            del x, full_x, w, s
        sum_a = torch.stack(left).double().sum(0).bfloat16()
        sum_b = torch.stack(right).double().sum(0).bfloat16()
        actual_a = records[0][0]["output"][0]
        actual_b = records[0][1]["output"][0][-1:]
        result = {"module_number": module_number, "ranks": per_rank,
                  "correctly_rounded_sums": row_metrics(sum_a, sum_b),
                  "decode_vs_fp64_sum": row_metrics(actual_a, sum_a),
                  "prefill_vs_fp64_sum": row_metrics(actual_b, sum_b),
                  "actual_decode_vs_prefill": row_metrics(actual_a, actual_b)}
        print(json.dumps(result), flush=True)
        results.append(result)
    assert results, "no wo_b snapshots found"
    with args.output.open("x") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
