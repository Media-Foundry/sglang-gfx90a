#!/usr/bin/env python3
"""Compute-only lower bound for TP4 M32 DSV4 attention-output row pipelining.

This intentionally does not wire a model selector.  It compares the production
math boundary

    wo_a: [M, G, D] x [G, R, D] -> [M, G, R]
    wo_b: [M, G*R] x [H, G*R]   -> [M, H]

with two independently scheduled row chunks.  The final TP all-reduce is not
included: both variants produce the same rank-local wo_b partial that would be
passed to that collective.
"""

from __future__ import annotations

import argparse
import statistics

import torch


def capture(fn):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def elapsed_us(graph, replay: int) -> float:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(replay):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / replay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--groups", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=4096)
    parser.add_argument("--rank", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--chunks", type=int, nargs="+", default=[2, 4, 8])
    parser.add_argument("--replay", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    torch.cuda.set_device(args.device)
    torch.manual_seed(20260829)
    dev = torch.device(f"cuda:{args.device}")
    x = torch.randn(
        args.rows, args.groups, args.head_dim, device=dev, dtype=torch.bfloat16
    )
    wa = torch.randn(
        args.groups, args.rank, args.head_dim, device=dev, dtype=torch.bfloat16
    ) / 64
    wb = torch.randn(
        args.hidden, args.groups * args.rank, device=dev, dtype=torch.bfloat16
    ) / 64
    out_a = torch.empty(args.rows, args.hidden, device=dev, dtype=torch.bfloat16)

    def baseline():
        mid = torch.einsum("tgd,grd->tgr", x, wa)
        torch.mm(mid.flatten(1), wb.t(), out=out_a)

    graph_a = capture(baseline)
    graph_a.replay()
    torch.cuda.synchronize()
    ref = out_a.clone()

    # The einsum is mathematically a strided batched GEMM over G.  Test that
    # direct spelling separately: transpose(0, 1) is a view, but restoring the
    # production [M,G,R] flattening for wo_b requires a contiguous materialize.
    out_bmm = torch.empty_like(out_a)

    def batched_bmm():
        mid_g = torch.bmm(x.transpose(0, 1), wa.transpose(1, 2))
        mid = mid_g.transpose(0, 1).contiguous()
        torch.mm(mid.flatten(1), wb.t(), out=out_bmm)

    graph_bmm = capture(batched_bmm)
    graph_bmm.replay()
    torch.cuda.synchronize()
    bmm_exact = torch.equal(ref, out_bmm)
    bmm_max_abs = float((ref.float() - out_bmm.float()).abs().max().item())

    candidates = []
    for chunks in args.chunks:
        if args.rows % chunks:
            continue
        chunk_rows = args.rows // chunks
        streams = [torch.cuda.Stream(device=dev) for _ in range(chunks)]
        done = [torch.cuda.Event() for _ in range(chunks)]
        out_b = torch.empty_like(out_a)

        def candidate():
            current = torch.cuda.current_stream()
            for idx, stream in enumerate(streams):
                lo = idx * chunk_rows
                hi = lo + chunk_rows
                stream.wait_stream(current)
                with torch.cuda.stream(stream):
                    mid = torch.einsum("tgd,grd->tgr", x[lo:hi], wa)
                    torch.mm(mid.flatten(1), wb.t(), out=out_b[lo:hi])
                    done[idx].record(stream)
            for event in done:
                current.wait_event(event)

        graph_b = capture(candidate)
        graph_b.replay()
        torch.cuda.synchronize()
        exact = torch.equal(ref, out_b)
        max_abs = float((ref.float() - out_b.float()).abs().max().item())
        candidates.append((chunks, graph_b, exact, max_abs))

    print(f"shape M={args.rows} G={args.groups} D={args.head_dim} R={args.rank} H={args.hidden}")
    print(f"correctness batched_bmm: exact={bmm_exact} max_abs={bmm_max_abs}")
    for chunks, _, exact, max_abs in candidates:
        print(f"correctness chunks={chunks}: exact={exact} max_abs={max_abs}")

    a_samples = []
    bmm_samples = []
    b_samples = {chunks: [] for chunks, _, _, _ in candidates}
    for round_idx in range(args.rounds):
        order = candidates if round_idx % 2 == 0 else list(reversed(candidates))
        a_samples.append(elapsed_us(graph_a, args.replay))
        bmm_samples.append(elapsed_us(graph_bmm, args.replay))
        for chunks, graph_b, _, _ in order:
            b_samples[chunks].append(elapsed_us(graph_b, args.replay))
        for chunks, graph_b, _, _ in reversed(order):
            b_samples[chunks].append(elapsed_us(graph_b, args.replay))
        a_samples.append(elapsed_us(graph_a, args.replay))
        bmm_samples.append(elapsed_us(graph_bmm, args.replay))

    a_med = statistics.median(a_samples)
    print(f"A full rows median_us={a_med:.3f} samples={','.join(f'{x:.3f}' for x in a_samples)}")
    bmm_med = statistics.median(bmm_samples)
    print(
        f"B batched_bmm median_us={bmm_med:.3f} "
        f"delta_pct={(a_med - bmm_med) / a_med * 100.0:+.2f} "
        f"exact={bmm_exact} max_abs={bmm_max_abs} samples="
        + ",".join(f"{x:.3f}" for x in bmm_samples)
    )
    for chunks, _, exact, max_abs in candidates:
        samples = b_samples[chunks]
        med = statistics.median(samples)
        delta = (a_med - med) / a_med * 100.0
        print(
            f"B chunks={chunks} median_us={med:.3f} delta_pct={delta:+.2f} "
            f"exact={exact} max_abs={max_abs} samples="
            + ",".join(f"{x:.3f}" for x in samples)
        )


if __name__ == "__main__":
    main()
