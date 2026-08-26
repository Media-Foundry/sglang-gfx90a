#!/usr/bin/env python3
"""ABBA CUDA-graph benchmark for gfx90a BF16 hipBLASLt solutions."""

import argparse
import csv
import statistics

import torch
import torch.nn.functional as F


def relative_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    delta = (actual.float() - expected.float()).square().sum().sqrt()
    scale = expected.float().square().sum().sqrt().clamp_min(1.0e-12)
    return float(delta / scale)


def graph_time_us(graph: torch.cuda.CUDAGraph, replays: int) -> float:
    for _ in range(20):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / replays


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--replays", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--fp32-output-n",
        default="2048,512",
        help="Comma-separated N values whose production consumer casts to FP32.",
    )
    args = parser.parse_args()
    fp32_output_n = {int(value) for value in args.fp32_output_n.split(",") if value}

    from aiter.tuned_gemm import hipb_gemm

    with open(args.csv_path, newline="") as handle:
        rows = list(csv.DictReader(handle))

    torch.manual_seed(17)
    for row in rows:
        if row.get("libtype") != "hipblaslt":
            continue
        m, n, k = (int(row[key]) for key in ("M", "N", "K"))
        solution = int(row["solidx"])
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        # Build hipBLASLt's process-global extension before graph capture.
        candidate_warm = hipb_gemm(x, weight, solution)
        reference_warm = F.linear(x, weight)
        torch.cuda.synchronize()

        def reference_fn():
            output = F.linear(x, weight)
            return output.float() if n in fp32_output_n else output

        def candidate_fn():
            output = hipb_gemm(x, weight, solution)
            return output.float() if n in fp32_output_n else output

        graph_a, reference = capture(reference_fn)
        graph_b, candidate = capture(candidate_fn)
        graph_a.replay()
        graph_b.replay()
        torch.cuda.synchronize()

        max_abs = float((candidate.float() - reference.float()).abs().max())
        rel_l2 = relative_l2(candidate, reference)
        finite = bool(torch.isfinite(candidate).all())
        warm_rel_l2 = relative_l2(candidate_warm, reference_warm)
        stable = True
        previous = candidate.clone()
        for _ in range(12):
            graph_b.replay()
            torch.cuda.synchronize()
            stable &= torch.equal(previous, candidate)
            previous.copy_(candidate)
        samples = []
        for _ in range(args.rounds):
            samples.extend(
                [
                    ("A", graph_time_us(graph_a, args.replays)),
                    ("B", graph_time_us(graph_b, args.replays)),
                    ("B", graph_time_us(graph_b, args.replays)),
                    ("A", graph_time_us(graph_a, args.replays)),
                ]
            )
        a_us = statistics.median(t for name, t in samples if name == "A")
        b_us = statistics.median(t for name, t in samples if name == "B")
        print(
            {
                "shape": [m, n, k],
                "solution": solution,
                "finite": finite,
                "replay_bitwise_stable": stable,
                "max_abs": max_abs,
                "relative_l2": rel_l2,
                "warm_relative_l2": warm_rel_l2,
                "torch_graph_us": a_us,
                "hipblaslt_graph_us": b_us,
                "speedup_percent": (a_us / b_us - 1.0) * 100.0,
            },
            flush=True,
        )


if __name__ == "__main__":
    main()
