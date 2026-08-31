#!/usr/bin/env python3
"""Compare DSpark-M128 compressor GEMMs with tuned hipBLASLt solutions.

This is deliberately a standalone oracle.  It does not change production
dispatch and must be run on one otherwise-idle gfx90a GCD.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from aiter.tuned_gemm import hipb_gemm, tgemm


SHAPES = (
    # M, N, K, tuned hipBLASLt solution
    (128, 2048, 4096, 4129),
    (128, 256, 4096, 5097),
)


def graph_callable(fn, x, w):
    for _ in range(10):
        fn(x, w)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn(x, w)
    return graph, out


def graph_us(graph, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    samples.sort()
    trim = max(1, repeats // 10)
    return statistics.mean(samples[trim:-trim])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    parser.add_argument("--timing-repeats", type=int, default=200)
    args = parser.parse_args()

    torch.manual_seed(20260901)
    torch.cuda.set_device(0)
    for m, n, k, solution in SHAPES:
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        w = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)

        def current(a, b):
            return tgemm.mm(a, b, otype=a.dtype)

        def candidate(a, b):
            return hipb_gemm(a, b, solution, otype=a.dtype)

        max_abs = 0.0
        max_rel_l2 = 0.0
        exact = 0
        for mutation in range(args.mutations):
            x.normal_(generator=None)
            if mutation % 8 == 0:
                w.normal_(generator=None)
            current_out = current(x, w)
            candidate_out = candidate(x, w)
            torch.cuda.synchronize()
            exact += int(torch.equal(current_out, candidate_out))
            max_abs = max(
                max_abs,
                float((current_out.float() - candidate_out.float()).abs().max()),
            )
            delta = current_out.float() - candidate_out.float()
            max_rel_l2 = max(
                max_rel_l2,
                float(delta.norm() / current_out.float().norm().clamp_min(1e-12)),
            )

        current_graph, current_out = graph_callable(current, x, w)
        candidate_graph, candidate_out = graph_callable(candidate, x, w)
        # Capture records work but does not guarantee that the captured GEMM
        # populated its output.  Establish the replay reference only after an
        # explicit first launch; otherwise this compares against uninitialized
        # graph-pool storage and falsely reports instability.
        candidate_graph.replay()
        torch.cuda.synchronize()
        expected = candidate_out.clone()
        stable = True
        replay_max_abs = 0.0
        for _ in range(args.graph_replays):
            candidate_graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(candidate_out, expected):
                stable = False
                replay_max_abs = max(
                    replay_max_abs,
                    float((candidate_out.float() - expected.float()).abs().max()),
                )
        torch.cuda.synchronize()

        # ABBA order limits clock/cache drift without mixing the graph captures.
        current_a = graph_us(current_graph, 50, args.timing_repeats)
        candidate_b = graph_us(candidate_graph, 50, args.timing_repeats)
        candidate_b2 = graph_us(candidate_graph, 50, args.timing_repeats)
        current_a2 = graph_us(current_graph, 50, args.timing_repeats)
        current_us = (current_a + current_a2) / 2.0
        candidate_us = (candidate_b + candidate_b2) / 2.0
        speedup = current_us / candidate_us
        print(
            f"M={m} N={n} K={k} solution={solution} "
            f"exact={exact}/{args.mutations} max_abs={max_abs:.8g} "
            f"max_rel_l2={max_rel_l2:.8g} graph_stable={stable} "
            f"replay_max_abs={replay_max_abs:.8g} current_us={current_us:.3f} "
            f"candidate_us={candidate_us:.3f} speedup={speedup:.4f}x"
        )


if __name__ == "__main__":
    main()
