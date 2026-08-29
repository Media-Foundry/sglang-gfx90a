#!/usr/bin/env python3
"""Compare production M32 MHC post/pre with the existing gfx90a full HIP path."""

from __future__ import annotations

import argparse
import statistics

import torch

import sglang.kernels.ops.layernorm.mhc as mhc_module
from sglang.kernels.ops.layernorm.gfx90a_mhc_post_pre import (
    gfx90a_mhc_post_pre,
)
from sglang.kernels.ops.layernorm.mhc import mhc_fused_post_pre


def capture(fn):
    result = fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fn()
    return graph, result


def elapsed(graph, warmup: int, iterations: int) -> float:
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


def error(expected: torch.Tensor, actual: torch.Tensor):
    diff = actual.float() - expected.float()
    return {
        "exact": torch.equal(expected, actual),
        "max_abs": float(diff.abs().max()),
        "rel_l2": float(
            torch.linalg.vector_norm(diff)
            / torch.linalg.vector_norm(expected.float())
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dir", default="/tmp/dsv4_ffn_dump.f3ZQ89")
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    args = parser.parse_args()

    prefix = f"{args.dump_dir}/layer_{args.layer}_rank_{args.rank}"

    def load(suffix: str) -> torch.Tensor:
        return torch.load(
            f"{prefix}_{suffix}.pt", map_location="cuda", weights_only=True
        ).contiguous()

    x = load("attn_out")
    residual = load("ffn_mhc_residual")
    post = load("ffn_mhc_post")
    comb = load("ffn_mhc_comb")
    fn = load("hc_ffn_fn")
    scale = load("hc_ffn_scale")
    base = load("hc_ffn_base")
    norm_weight = load("ffn_norm_weight").bfloat16().contiguous()
    if x.shape != (32, 4096) or residual.shape != (32, 4, 4096):
        raise RuntimeError(f"expected M32 dump, got x={x.shape}, residual={residual.shape}")
    fn_fp16 = fn.half().contiguous()
    # This is a single-GCD component oracle.  The production wrapper enters a
    # disabled symmetric-memory context even though this path launches no
    # collective; avoid requiring an initialized TP process group here.
    mhc_module.get_tp_group = lambda: None
    mhc_module.is_allocation_symmetric = lambda: False

    def baseline():
        return mhc_fused_post_pre(
            x,
            residual,
            post,
            comb,
            fn,
            scale,
            base,
            1e-6,
            1e-6,
            1e-6,
            2.0,
            20,
            n_splits=8,
            tile_n=1,
            norm_weight=norm_weight,
            norm_eps=1e-6,
            global_batch_size=32,
            fn_fp16=fn_fp16,
        )

    def candidate():
        result = gfx90a_mhc_post_pre(
            x,
            residual,
            post,
            comb,
            fn_fp16,
            scale,
            base,
            norm_weight,
            1e-6,
            1e-6,
            2.0,
            1e-6,
        )
        if result is None:
            raise RuntimeError("native MHC rejected the real M32 shape")
        return result

    graph_a, out_a = capture(baseline)
    graph_b, out_b = capture(candidate)
    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()
    print("correctness", [error(a, b) for a, b in zip(out_a, out_b)], flush=True)

    samples_a, samples_b = [], []
    for _ in range(args.rounds):
        samples_a.append(elapsed(graph_a, args.warmup, args.iterations))
        samples_b.append(elapsed(graph_b, args.warmup, args.iterations))
        samples_b.append(elapsed(graph_b, args.warmup, args.iterations))
        samples_a.append(elapsed(graph_a, args.warmup, args.iterations))
    median_a = statistics.median(samples_a)
    median_b = statistics.median(samples_b)
    print(
        {
            "baseline_us": median_a,
            "native_us": median_b,
            "delta_us": median_a - median_b,
            "delta_pct": (median_a - median_b) / median_a * 100.0,
            "baseline_samples": samples_a,
            "native_samples": samples_b,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
