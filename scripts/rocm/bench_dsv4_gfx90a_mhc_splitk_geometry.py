#!/usr/bin/env python3
"""Sweep the real M32 DSV4 MHC split-K pre-mix geometry on gfx90a."""

from __future__ import annotations

import argparse
import itertools
import statistics
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_mix_splitk_stage0_kernel,
)


@triton.jit
def _splitk_tail_kernel(
    dot_partials,
    rms_partials,
    residual,
    hc_scale,
    hc_base,
    norm_weight,
    post,
    comb,
    out,
    SPLITS: tl.constexpr,
    eps: tl.constexpr,
    norm_eps: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    """Production fused tail with only its hard-coded split extent parameterized."""
    token_id = tl.program_id(0)
    splits = tl.arange(0, SPLITS)
    rms_offsets = tl.arange(0, 64)
    sq_sum = tl.sum(tl.load(rms_partials + token_id * 64 + rms_offsets))
    pre_scale = tl.rsqrt(sq_sum / 16384.0 + eps)

    offs4 = tl.arange(0, 4)
    pre_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + offs4[:, None]) * SPLITS
            + splits[None, :]
        ),
        axis=1,
    )
    post_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + 4 + offs4[:, None]) * SPLITS
            + splits[None, :]
        ),
        axis=1,
    )
    pre_v = tl.sigmoid(
        pre_dot * pre_scale * tl.load(hc_scale) + tl.load(hc_base + offs4)
    ) + eps
    post_v = 2.0 * tl.sigmoid(
        post_dot * pre_scale * tl.load(hc_scale + 1)
        + tl.load(hc_base + 4 + offs4)
    )
    tl.store(post + token_id * 4 + offs4, post_v)

    offs16 = tl.arange(0, 16)
    comb_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + 8 + offs16[:, None]) * SPLITS
            + splits[None, :]
        ),
        axis=1,
    )
    matrix = tl.reshape(
        comb_dot * pre_scale * tl.load(hc_scale + 2)
        + tl.load(hc_base + 8 + offs16),
        (4, 4),
    )
    matrix = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = matrix / tl.sum(matrix, axis=1)[:, None] + eps
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    for _ in tl.static_range(0, SINKHORN_ITERS - 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + eps)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    tl.store(comb + token_id * 16 + offs16, tl.reshape(matrix, (16,)))

    hidden = tl.arange(0, 4096)
    channels = tl.arange(0, 4)
    residual_v = tl.load(
        residual
        + token_id * 16384
        + channels[:, None] * 4096
        + hidden[None, :]
    ).to(tl.float32)
    weighted = tl.sum(pre_v[:, None] * residual_v, axis=0)
    rounded = weighted.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(rounded * rounded) / 4096.0 + norm_eps)
    weight = tl.load(norm_weight + hidden).to(tl.float32)
    tl.store(out + token_id * 4096 + hidden, rounded * inv_rms * weight)


@dataclass
class Geometry:
    block_n: int
    splits: int
    block_k: int

    def label(self) -> str:
        return f"N{self.block_n}/S{self.splits}/K{self.block_k}"


class Runner:
    def __init__(
        self,
        geometry: Geometry,
        residual: torch.Tensor,
        fn: torch.Tensor,
        rms_partials: torch.Tensor,
        hc_scale: torch.Tensor,
        hc_base: torch.Tensor,
        norm_weight: torch.Tensor,
        sinkhorn_eps: float,
        norm_eps: float,
        sinkhorn_iters: int,
    ) -> None:
        self.g = geometry
        self.residual = residual
        self.fn = fn
        self.rms_partials = rms_partials
        self.hc_scale = hc_scale
        self.hc_base = hc_base
        self.norm_weight = norm_weight
        self.sinkhorn_eps = sinkhorn_eps
        self.norm_eps = norm_eps
        self.sinkhorn_iters = sinkhorn_iters
        m = residual.shape[0]
        self.dot = torch.empty((m, 24, geometry.splits), device=residual.device)
        self.post = torch.empty((m, 4), dtype=torch.float32, device=residual.device)
        self.comb = torch.empty((m, 4, 4), dtype=torch.float32, device=residual.device)
        self.out = torch.empty((m, 4096), dtype=torch.bfloat16, device=residual.device)

    def run(self):
        g = self.g
        m = self.residual.shape[0]
        _gfx90a_mhc_mix_splitk_stage0_kernel[
            (triton.cdiv(24, g.block_n), g.splits, m)
        ](
            self.residual.flatten(1),
            self.fn,
            self.dot,
            k=16384,
            SPLITS=g.splits,
            CHUNK_K=16384 // g.splits,
            BLOCK_N=g.block_n,
            BLOCK_K=g.block_k,
            num_warps=1,
        )
        _splitk_tail_kernel[(m,)](
            self.dot,
            self.rms_partials,
            self.residual,
            self.hc_scale,
            self.hc_base,
            self.norm_weight,
            self.post,
            self.comb,
            self.out,
            SPLITS=g.splits,
            eps=self.sinkhorn_eps,
            norm_eps=self.norm_eps,
            SINKHORN_ITERS=self.sinkhorn_iters,
            num_warps=8,
        )
        return self.post, self.comb, self.out

    def capture(self) -> torch.cuda.CUDAGraph:
        self.run()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self.run()
        return graph


def timed_graph(graph: torch.cuda.CUDAGraph, warmup: int, iterations: int) -> float:
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


def abba(graph_a, graph_b, warmup: int, iterations: int, rounds: int):
    a, b = [], []
    for _ in range(rounds):
        a.append(timed_graph(graph_a, warmup, iterations))
        b.append(timed_graph(graph_b, warmup, iterations))
        b.append(timed_graph(graph_b, warmup, iterations))
        a.append(timed_graph(graph_a, warmup, iterations))
    return a, b


def errors(ref: torch.Tensor, got: torch.Tensor) -> tuple[bool, float, float]:
    rf, gf = ref.float(), got.float()
    diff = gf - rf
    rel_l2 = float(torch.linalg.vector_norm(diff) / torch.linalg.vector_norm(rf))
    return torch.equal(ref, got), float(diff.abs().max()), rel_l2


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump-dir", default="/tmp/dsv4_ffn_dump.f3ZQ89")
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--sinkhorn-eps", type=float, default=1e-6)
    p.add_argument("--norm-eps", type=float, default=1e-6)
    p.add_argument("--sinkhorn-iters", type=int, default=8)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    args = p.parse_args()
    if not torch.version.hip:
        raise RuntimeError("ROCm required")

    device = torch.device("cuda")
    prefix = f"{args.dump_dir}/layer_{args.layer}_rank_{args.rank}"
    load = lambda suffix: torch.load(
        f"{prefix}_{suffix}.pt", map_location=device, weights_only=True
    ).contiguous()
    residual = load("ffn_mhc_residual")
    fn = load("hc_ffn_fn")
    hc_scale = load("hc_ffn_scale")
    hc_base = load("hc_ffn_base")
    norm_weight = load("ffn_norm_weight")
    if residual.shape != (32, 4, 4096):
        raise RuntimeError(f"expected real M32 residual, got {residual.shape}")

    # The tail consumes 64 FP32 RMS partials. Preserve its exact consumer shape;
    # only split-K dot geometry is under test.
    x = residual.float().flatten(1)
    rms_partials = (x.square().reshape(32, 64, 256).sum(-1)).contiguous()
    baseline_g = Geometry(4, 8, 1024)
    baseline = Runner(
        baseline_g, residual, fn, rms_partials, hc_scale, hc_base, norm_weight,
        args.sinkhorn_eps, args.norm_eps, args.sinkhorn_iters,
    )
    baseline.run()
    torch.cuda.synchronize()
    ref = tuple(t.clone() for t in (baseline.post, baseline.comb, baseline.out))
    baseline_graph = baseline.capture()

    results = []
    geometries = [
        Geometry(n, s, k)
        for n, s, k in itertools.product((1, 2, 4, 8), (4, 8, 16), (512, 1024, 2048))
    ]
    for g in geometries:
        candidate = Runner(
            g, residual, fn, rms_partials, hc_scale, hc_base, norm_weight,
            args.sinkhorn_eps, args.norm_eps, args.sinkhorn_iters,
        )
        candidate.run()
        torch.cuda.synchronize()
        corr = [errors(r, c) for r, c in zip(ref, (candidate.post, candidate.comb, candidate.out))]
        graph = candidate.capture()
        samples_a, samples_b = abba(
            baseline_graph, graph, args.warmup, args.iterations, args.rounds
        )
        a, b = statistics.median(samples_a), statistics.median(samples_b)
        saved = a - b
        results.append((saved, g, corr, a, b, samples_a, samples_b))
        print(
            f"geometry={g.label()} base_us={a:.3f} candidate_us={b:.3f} "
            f"saved_us={saved:+.3f} delta_pct={(b/a-1)*100:+.2f} "
            f"post={corr[0]} comb={corr[1]} out={corr[2]}", flush=True
        )

    print("\nranked_by_saved_us", flush=True)
    for saved, g, corr, a, b, sa, sb in sorted(results, reverse=True, key=lambda v: v[0]):
        print(
            f"{g.label():14s} saved_us={saved:+.3f} base_us={a:.3f} candidate_us={b:.3f} "
            f"all_exact={all(v[0] for v in corr)} "
            f"base_samples={[round(v,3) for v in sa]} candidate_samples={[round(v,3) for v in sb]}",
            flush=True,
        )


if __name__ == "__main__":
    main()
