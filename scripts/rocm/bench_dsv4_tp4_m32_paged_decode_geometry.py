#!/usr/bin/env python3
"""Oracle-only unified-KV TP4 M32/H16/D512 decode geometry sweep."""

import argparse
import statistics

import torch

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)


PROFILES = tuple((w, s) for w in (2, 4, 8) for s in (1, 2, 3))
BASELINE = (4, 2)


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def time_graph(graph, warmup, iterations):
    for _ in range(warmup): graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations): graph.replay()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed(values):
    values = sorted(values)
    return statistics.mean(values[1:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--contexts", default="256,512,768,1024")
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--mutations", type=int, default=100)
    args = p.parse_args()
    torch.manual_seed(20260830)

    for context in (int(x) for x in args.contexts.split(",")):
        t, h, d = 32, 16, 512
        q = torch.randn((t, h, d), dtype=torch.bfloat16, device="cuda")
        kv = torch.randn((t * context, d), dtype=torch.bfloat16, device="cuda")
        indices = torch.arange(t * context, dtype=torch.int32, device="cuda")
        indptr = torch.arange(0, (t + 1) * context, context,
                              dtype=torch.int32, device="cuda")
        sink = torch.randn((h,), dtype=torch.float32, device="cuda")

        def run(profile):
            warps, stages = profile
            return _sparse_attn_v4_paged_decode_triton(
                q, kv, indices, indptr, sink, 1.0 / (d ** 0.5),
                block_h=16, kv_splits=4, block_k=16,
                _oracle_num_warps=warps,
                _oracle_num_stages=stages,
            )

        graphs, outputs = {}, {}
        for profile in PROFILES:
            graphs[profile], outputs[profile] = capture(lambda p=profile: run(p))
        reference = outputs[BASELINE]
        for profile, output in outputs.items():
            if not torch.equal(reference, output):
                delta = (reference.float() - output.float()).abs().max().item()
                raise RuntimeError(
                    f"ctx={context} profile={profile} mismatch max_abs={delta}"
                )
        print(f"CORRECTNESS ctx={context} profiles={len(PROFILES)} bitwise_exact=True")

        for mutation in range(args.mutations):
            q.normal_()
            graphs[BASELINE].replay(); graphs[(2, 1)].replay()
            torch.cuda.synchronize()
            if not torch.equal(outputs[BASELINE], outputs[(2, 1)]):
                delta = (outputs[BASELINE].float() - outputs[(2, 1)].float()).abs().max().item()
                raise RuntimeError(
                    f"ctx={context} mutation={mutation} best mismatch max_abs={delta}"
                )
        print(f"MUTATIONS ctx={context} count={args.mutations} best_bitwise_exact=True")

        values = {profile: [] for profile in PROFILES}
        order = list(PROFILES)
        for _ in range(args.rounds):
            for profile in order + list(reversed(order)):
                values[profile].append(
                    time_graph(graphs[profile], 10, args.iterations)
                )
        summary = {profile: trimmed(samples) for profile, samples in values.items()}
        base = summary[BASELINE]
        for profile in PROFILES:
            value = summary[profile]
            print(
                f"RESULT ctx={context} warps={profile[0]} stages={profile[1]} "
                f"us={value:.3f} saving_us={base-value:.3f} "
                f"gain_pct={(base/value-1)*100:.3f}"
            )
        best = min(summary, key=summary.get); value = summary[best]
        print(
            f"DECISION ctx={context} baseline_us={base:.3f} best={best} "
            f"best_us={value:.3f} pass={(base-value)>=5.0 or (base/value-1)>=0.10}"
        )


if __name__ == "__main__": main()
