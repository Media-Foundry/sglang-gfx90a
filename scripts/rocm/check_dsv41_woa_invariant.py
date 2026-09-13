#!/usr/bin/env python3
"""Adversarial row-placement and graph replay checks for the fixed wo_a."""

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
    from sglang.kernels.ops.attention.dsv4.wo_a_bf16_invariant import wo_a_bf16_invariant
    candidates = []
    for f in args.trace_dir.glob("rank1-*MQALayer*-call0-*.pt"):
        r = torch.load(f, map_location="cpu", weights_only=True)
        if r["layer_id"] == 6:
            candidates.append(r["attention_contract"]["wo_a_weight"])
    assert len(candidates) == 1
    weight = candidates[0].cuda()
    torch.manual_seed(41366)
    report = {"layer": 6, "rank": 1, "cases": [], "mutation_exact": 0, "replay_exact": 0}
    for groups in (1, 2):
        w = weight.repeat(groups, 1, 1)
        for m in (1, 2, 3, 15, 16, 17, 32, 64, 128, 203, 204, 256):
            # Match non-contiguous local-head storage and vary all other rows.
            padded = torch.randn(m, 8, 4096, dtype=torch.bfloat16, device="cuda")
            x = padded[:, :groups]
            y = wo_a_bf16_invariant(x, w)
            positions = sorted({0, m // 2, m - 1})
            for pos in positions:
                single = wo_a_bf16_invariant(x[pos:pos + 1], w)
                metrics = row_metrics(y[pos:pos + 1].cpu(), single.cpu())
                reference = torch.einsum("tgd,grd->tgr", x[pos:pos + 1].cpu().double(), w.cpu().double()).bfloat16()
                ref_metrics = row_metrics(single.cpu(), reference)
                assert metrics["exact"] and ref_metrics["finite"] and ref_metrics["relative_l2"] < 0.001
                report["cases"].append({"m": m, "g": groups, "row": pos, "alignment": metrics, "fp64": ref_metrics})
    x = torch.randn(17, 8, 4096, dtype=torch.bfloat16, device="cuda")[:, :1]
    wo_a_bf16_invariant(x, weight)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = wo_a_bf16_invariant(x, weight)
    for _ in range(100):
        x.copy_(torch.randn_like(x))
        graph.replay()
        eager = wo_a_bf16_invariant(x, weight)
        assert torch.equal(captured, eager)
        report["mutation_exact"] += 1
    expected = captured.clone()
    checks = torch.empty(1000, dtype=torch.bool, device="cuda")
    for i in range(1000):
        graph.replay()
        checks[i] = torch.all(captured == expected)
    report["replay_exact"] = int(checks.sum().item())
    assert report["replay_exact"] == 1000
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
    print(json.dumps({"cases": len(report["cases"]), "mutation_exact": report["mutation_exact"],
                      "replay_exact": report["replay_exact"]}))


if __name__ == "__main__":
    main()
