#!/usr/bin/env python3
"""M32 gfx90a oracle for a segmented ``wqkv_a -> q RMSNorm`` producer.

This intentionally stays outside the model selector.  It answers the first
question required by a CK-style fused producer: can splitting the logical
``N=1536`` projection into its ``q_lora=1024`` and ``kv=512`` segments leave
enough latency headroom to fuse the q RMSNorm?  The candidate is an optimistic
lower bound: it uses the installed BLAS for both segments and the production
AIter RMSNorm, without charging any CK workspace/finalization overhead.

If this lower bound does not beat the production-shape projection plus RMSNorm
by 10%, a custom segmented CK epilogue cannot be justified.
"""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


Q_WIDTH = 1024
KV_WIDTH = 512
HIDDEN = 4096


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    return parser.parse_args()


def load_inputs(args: argparse.Namespace):
    prefix = args.dump_dir / f"layer_{args.layer}_rank_{args.rank}"
    x = torch.load(
        Path(f"{prefix}_attn_norm.pt"), map_location="cpu", weights_only=True
    )
    weight = torch.load(
        Path(f"{prefix}_projection_wqkv_a.pt"),
        map_location="cpu",
        weights_only=True,
    )
    key = f"layers.{args.layer}.attn.q_norm.weight"
    import json

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    shard = args.model_dir / index[key]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        norm_weight = handle.get_tensor(key)
    if x.shape != (32, HIDDEN) or x.dtype != torch.bfloat16:
        raise ValueError(f"expected BF16 [32,{HIDDEN}] input, got {x.shape} {x.dtype}")
    if weight.shape != (Q_WIDTH + KV_WIDTH, HIDDEN) or weight.dtype != torch.bfloat16:
        raise ValueError(f"unexpected wqkv weight {weight.shape} {weight.dtype}")
    if norm_weight.shape != (Q_WIDTH,):
        raise ValueError(f"unexpected q_norm weight {norm_weight.shape}")
    return x.contiguous(), weight.contiguous(), norm_weight.contiguous()


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def graph_time_us(graph: torch.cuda.CUDAGraph, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iters


def metrics(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
    delta = actual.float() - expected.float()
    denom = expected.float().square().sum().sqrt().clamp_min(1.0e-12)
    return {
        "exact": bool(torch.equal(actual, expected)),
        "max_abs": float(delta.abs().max()),
        "relative_l2": float(delta.square().sum().sqrt() / denom),
    }


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(0)
    x_cpu, weight_cpu, norm_weight_cpu = load_inputs(args)
    x = x_cpu.cuda()
    weight = weight_cpu.cuda()
    norm_weight = norm_weight_cpu.cuda().to(torch.bfloat16)
    q_weight, kv_weight = weight.split((Q_WIDTH, KV_WIDTH), dim=0)

    from aiter import rmsnorm2d_fwd

    eps = 1.0e-6

    def reference():
        raw = F.linear(x, weight)
        q = rmsnorm2d_fwd(raw[:, :Q_WIDTH], norm_weight, eps)
        return raw, q

    def split_candidate():
        q_raw = F.linear(x, q_weight)
        kv = F.linear(x, kv_weight)
        q = rmsnorm2d_fwd(q_raw, norm_weight, eps)
        return q_raw, kv, q

    # First-use library initialization must stay outside graph capture.
    ref_warm = reference()
    split_warm = split_candidate()
    torch.cuda.synchronize()
    graph_a, ref = capture(reference)
    graph_b, split = capture(split_candidate)
    graph_projection, _ = capture(lambda: F.linear(x, weight))
    raw_for_norm = F.linear(x, weight)[:, :Q_WIDTH].contiguous()
    torch.cuda.synchronize()
    graph_norm, _ = capture(
        lambda: rmsnorm2d_fwd(raw_for_norm, norm_weight, eps)
    )
    graph_a.replay()
    graph_b.replay()
    torch.cuda.synchronize()

    raw_ref, q_ref = ref
    q_raw, kv, q_split = split
    print(
        {
            "raw_q": metrics(q_raw, raw_ref[:, :Q_WIDTH]),
            "raw_kv": metrics(kv, raw_ref[:, Q_WIDTH:]),
            "normalized_q": metrics(q_split, q_ref),
            "warm_normalized_q": metrics(split_warm[2], ref_warm[1]),
        },
        flush=True,
    )

    stable = True
    previous = tuple(t.clone() for t in split)
    for _ in range(20):
        graph_b.replay()
        torch.cuda.synchronize()
        stable &= all(torch.equal(a, b) for a, b in zip(previous, split, strict=True))
        for dst, src in zip(previous, split, strict=True):
            dst.copy_(src)

    samples: list[tuple[str, float]] = []
    for _ in range(args.rounds):
        samples.append(("A", graph_time_us(graph_a, args.warmup, args.iters)))
        samples.append(("B", graph_time_us(graph_b, args.warmup, args.iters)))
        samples.append(("B", graph_time_us(graph_b, args.warmup, args.iters)))
        samples.append(("A", graph_time_us(graph_a, args.warmup, args.iters)))
    a = [value for label, value in samples if label == "A"]
    b = [value for label, value in samples if label == "B"]
    a_median, b_median = statistics.median(a), statistics.median(b)
    projection_us = [
        graph_time_us(graph_projection, args.warmup, args.iters)
        for _ in range(args.rounds)
    ]
    norm_us = [
        graph_time_us(graph_norm, args.warmup, args.iters)
        for _ in range(args.rounds)
    ]
    projection_median = statistics.median(projection_us)
    norm_median = statistics.median(norm_us)
    print(
        {
            "replay_bitwise_stable": stable,
            "A_us": a,
            "B_split_lower_bound_us": b,
            "A_median_us": a_median,
            "B_median_us": b_median,
            "speedup_percent": (a_median / b_median - 1.0) * 100.0,
            "continue_gate_percent": 10.0,
            "passes_continue_gate": b_median <= 0.9 * a_median,
            "projection_only_us": projection_us,
            "projection_only_median_us": projection_median,
            "rmsnorm_only_us": norm_us,
            "rmsnorm_only_median_us": norm_median,
            "ideal_remove_norm_speedup_percent": (
                a_median / projection_median - 1.0
            )
            * 100.0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
