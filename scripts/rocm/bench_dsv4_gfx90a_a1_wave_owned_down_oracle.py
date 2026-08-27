#!/usr/bin/env python3
"""A1 wave-owned singleton-expert down oracle for TP8 M32 on gfx90a."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_a1_wave_owned_oracle import (
    _jit_a1_wave_owned_down,
)
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_grouped


E, M, T, N, K = 256, 32, 6, 4096, 256


def reconstruct_topk(counts: torch.Tensor) -> torch.Tensor:
    rows: list[list[int]] = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            candidates = [r for r in range(M) if len(rows[r]) < T and expert not in rows[r]]
            if not candidates:
                raise RuntimeError(f"cannot place expert {expert}")
            row = min(candidates, key=lambda r: (len(rows[r]), r))
            rows[row].append(expert)
    result = torch.tensor(rows, dtype=torch.int32)
    if result.shape != (M, T):
        raise RuntimeError(f"bad top-k shape {result.shape}")
    return result


def a1_metadata(topk: torch.Tensor, device: torch.device):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, experts in enumerate(topk.tolist()):
        for slot, expert in enumerate(experts):
            buckets[expert].append((slot << 24) | token)
    ids, experts = [], []
    for expert, bucket in enumerate(buckets):
        if len(bucket) == 1:
            ids.append(bucket[0])
            experts.append(expert)
    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


def capture(fn) -> torch.cuda.CUDAGraph:
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def timed_graph(graph, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def abba(a, b, warmup: int, iterations: int, rounds: int):
    sa, sb = [], []
    for _ in range(rounds):
        sa.append(timed_graph(a, warmup, iterations))
        sb.append(timed_graph(b, warmup, iterations))
        sb.append(timed_graph(b, warmup, iterations))
        sa.append(timed_graph(a, warmup, iterations))
    return sa, sb


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recorder", required=True)
    p.add_argument("--pass-index", type=int, default=37)
    p.add_argument("--layer", type=int, default=34)
    p.add_argument("--world-size", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--correctness-replays", type=int, default=100)
    args = p.parse_args()
    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    device = torch.device("cuda")
    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    raw = payload["logical_count"][args.pass_index, args.layer]
    if torch.any(raw.remainder(args.world_size) != 0):
        raise RuntimeError("counts are not divisible by TP world size")
    counts = raw // args.world_size
    topk = reconstruct_topk(counts)
    ids, experts, valid = a1_metadata(topk, device)
    print(
        f"routing pass={args.pass_index} layer={args.layer} "
        f"active={int((counts > 0).sum())} max_occ={int(counts.max())} "
        f"a1_experts={experts.numel()}", flush=True,
    )
    if experts.numel() == 0:
        raise RuntimeError("selected layer has no A1 experts")

    torch.manual_seed(7)
    iq = torch.randint(-127, 128, (M, T, K), dtype=torch.int8, device=device)
    isc = torch.rand((M, T, K // 32), dtype=torch.float32, device=device) / 64
    w2 = torch.randint(0, 256, (E, N, K // 2), dtype=torch.uint8, device=device)
    s2 = torch.full((E, N, K // 32), 127, dtype=torch.uint8, device=device)
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    out_ref = torch.zeros((M, T, N), dtype=torch.float32, device=device)
    out_candidate = torch.zeros_like(out_ref)

    results = []
    for blocks in (104, 208, 416, 832):
        ref_module = _jit_down_grouped(E, M, T, N, K, 1, 2, 8, blocks, 2)
        candidate_module = _jit_a1_wave_owned_down(blocks)

        def run_ref():
            ref_module.run_partial(
                iq, isc, w2, s2, ids, experts, valid, topk_weights, out_ref
            )

        def run_candidate():
            candidate_module.run(
                iq, isc, w2, s2, ids, experts, valid, topk_weights, out_candidate
            )

        run_ref()
        run_candidate()
        torch.cuda.synchronize()
        exact = torch.equal(out_ref, out_candidate)
        max_abs = float((out_ref - out_candidate).abs().max())
        print(
            f"correctness blocks={blocks} partial_exact={exact} max_abs={max_abs:.8g}",
            flush=True,
        )
        if not exact:
            raise AssertionError(f"blocks={blocks} is not bitwise exact")

        if blocks == 832:
            for replay in range(args.correctness_replays):
                iq.add_((replay % 5) + 1)
                run_ref()
                run_candidate()
                torch.cuda.synchronize()
                if not torch.equal(out_ref, out_candidate):
                    diff = float((out_ref - out_candidate).abs().max())
                    raise AssertionError(
                        f"mutation={replay} mismatch max_abs={diff:.8g}"
                    )
            print(
                f"correctness_mutations={args.correctness_replays} "
                "partial_exact=True", flush=True,
            )

        graph_ref = capture(run_ref)
        graph_candidate = capture(run_candidate)
        sa, sb = abba(
            graph_ref, graph_candidate, args.warmup, args.iterations, args.rounds
        )
        ma, mb = statistics.median(sa), statistics.median(sb)
        results.append((ma - mb, blocks, ma, mb, sa, sb))
        print(
            f"ABBA blocks={blocks} reference_us={ma:.3f} candidate_us={mb:.3f} "
            f"saved_us={ma-mb:+.3f} delta_pct={(mb/ma-1)*100:+.2f} "
            f"reference_samples={[round(v,3) for v in sa]} "
            f"candidate_samples={[round(v,3) for v in sb]}", flush=True,
        )

    print("ranked", flush=True)
    for saved, blocks, ma, mb, _, _ in sorted(results, reverse=True):
        print(
            f"blocks={blocks} saved_us={saved:+.3f} "
            f"reference_us={ma:.3f} candidate_us={mb:.3f}", flush=True,
        )


if __name__ == "__main__":
    main()
