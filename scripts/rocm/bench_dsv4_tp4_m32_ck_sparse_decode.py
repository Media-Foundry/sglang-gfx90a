#!/usr/bin/env python3
"""ABBA oracle for the gfx90a TP4 M32 CK-style sparse decode candidate."""

import argparse
import statistics

import torch

from sglang.kernels.ops.attention.dsv4.gfx90a_unified_sparse_decode import (
    run as run_ck,
    workspace_size_bytes,
)
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
)


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    torch.cuda.synchronize()
    return graph, out


def time_graph(graph, warmup, iterations):
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed(values):
    ordered = sorted(values)
    return statistics.mean(ordered[1:-1])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="128,256,512")
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--mutations", type=int, default=100)
    parser.add_argument("--max-abs", type=float, default=0.008)
    parser.add_argument("--max-rel-l2", type=float, default=0.005)
    args = parser.parse_args()

    torch.manual_seed(20260831)
    tokens, heads, dim = 32, 16, 512
    for context in (int(value) for value in args.contexts.split(",")):
        q = torch.randn((tokens, heads, dim), dtype=torch.bfloat16, device="cuda")
        kv = torch.randn(
            (tokens * context, dim), dtype=torch.bfloat16, device="cuda"
        )
        indices = torch.arange(
            tokens * context, dtype=torch.int32, device="cuda"
        )
        indptr = torch.arange(
            0,
            (tokens + 1) * context,
            context,
            dtype=torch.int32,
            device="cuda",
        )
        sink = torch.randn((heads,), dtype=torch.float32, device="cuda")
        scale = 1.0 / (dim**0.5)

        def triton_run():
            return _sparse_attn_v4_paged_decode_triton(
                q,
                kv,
                indices,
                indptr,
                sink,
                scale,
                block_h=16,
                kv_splits=4,
                block_k=16,
                _oracle_num_warps=4,
                _oracle_num_stages=2,
            )

        ck_out = torch.empty_like(q)
        workspace = torch.empty(
            workspace_size_bytes(tokens=tokens), dtype=torch.uint8, device="cuda"
        )

        def ck_run():
            run_ck(q, kv, indices, indptr, sink, ck_out, workspace, scale)
            return ck_out

        # Validate eagerly.  Keeping independently captured Triton allocation
        # pools alive across context sizes can alias a subsequently captured
        # output on ROCm and create a benchmark-only stale-output artifact.
        triton_out = triton_run()
        ck_run()
        torch.cuda.synchronize()
        max_abs = (triton_out.float() - ck_out.float()).abs().max().item()
        rel_l2 = (
            torch.linalg.vector_norm(triton_out.float() - ck_out.float())
            / torch.linalg.vector_norm(triton_out.float()).clamp_min(1.0e-12)
        ).item()
        if max_abs > args.max_abs or rel_l2 > args.max_rel_l2:
            raise RuntimeError(
                f"ctx={context} initial max_abs={max_abs} rel_l2={rel_l2}"
            )

        mutation_max_abs = max_abs
        mutation_max_rel_l2 = rel_l2
        for mutation in range(args.mutations):
            q.normal_()
            triton_out = triton_run()
            ck_run()
            torch.cuda.synchronize()
            delta = (triton_out.float() - ck_out.float()).abs().max().item()
            delta_rel_l2 = (
                torch.linalg.vector_norm(triton_out.float() - ck_out.float())
                / torch.linalg.vector_norm(triton_out.float()).clamp_min(1.0e-12)
            ).item()
            mutation_max_abs = max(mutation_max_abs, delta)
            mutation_max_rel_l2 = max(mutation_max_rel_l2, delta_rel_l2)
            if delta > args.max_abs or delta_rel_l2 > args.max_rel_l2:
                raise RuntimeError(
                    f"ctx={context} mutation={mutation} max_abs={delta} "
                    f"rel_l2={delta_rel_l2}"
                )
        print(
            f"CORRECTNESS ctx={context} mutations={args.mutations} "
            f"max_abs={mutation_max_abs:.9f} "
            f"max_rel_l2={mutation_max_rel_l2:.9f}"
        )

        triton_graph, _ = capture(triton_run)
        ck_graph, _ = capture(ck_run)
        values = {"triton": [], "ck": []}
        for _ in range(args.rounds):
            for provider in ("triton", "ck", "ck", "triton"):
                graph = triton_graph if provider == "triton" else ck_graph
                values[provider].append(time_graph(graph, 10, args.iterations))
        triton_us = trimmed(values["triton"])
        ck_us = trimmed(values["ck"])
        print(
            f"RESULT ctx={context} triton_us={triton_us:.3f} ck_us={ck_us:.3f} "
            f"saving_us={triton_us-ck_us:.3f} "
            f"gain_pct={(triton_us/ck_us-1)*100:.3f}"
        )


if __name__ == "__main__":
    main()
