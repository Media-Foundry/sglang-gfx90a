#!/usr/bin/env python3
"""Lower-bound oracle for fusing DSV4 inverse RoPE into TP8 ``wo_a``.

No fused kernel is implemented here.  The oracle measures the maximum exposed
latency that a future CK/HIP A-load transform could remove:

* A: copy raw attention output -> production inverse RoPE -> production wo_a;
* G: copy already-rotated output -> production wo_a;
* R: copy raw attention output -> production inverse RoPE;
* C: copy only (common graph-replay reset cost).

The real layer-20 M32 dump and the checkpoint's FP8 wo_a+E8M0 scale are used.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
from safetensors import safe_open


M = 32
HEADS = 8
HEAD_DIM = 512
ROPE_DIM = 64
GROUPS = 1
RANK_OUT = 1024


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


def load_tensor(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True).contiguous()


def load_runtime_woa_shard(args: argparse.Namespace) -> torch.Tensor:
    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    weight_key = f"layers.{args.layer}.attn.wo_a.weight"
    scale_key = f"layers.{args.layer}.attn.wo_a.scale"
    weight_shard = args.model_dir / index[weight_key]
    if index[scale_key] != index[weight_key]:
        raise ValueError("wo_a weight and scale unexpectedly live in different shards")
    with safe_open(weight_shard, framework="pt", device="cpu") as handle:
        packed = handle.get_tensor(weight_key)
        scale = handle.get_tensor(scale_key)
    if packed.shape != (8192, 4096) or packed.dtype != torch.float8_e4m3fn:
        raise ValueError(f"unexpected wo_a packed tensor {packed.shape} {packed.dtype}")
    if scale.shape != (64, 32):
        raise ValueError(f"unexpected wo_a scale {scale.shape} {scale.dtype}")
    # Match DeepSeekV4 `_dequant_fp8`: scales cover [128,128] blocks and the
    # runtime rounds the dequantized parameter to BF16 before TP slicing.
    logical = (
        packed.float().view(64, 128, 32, 128)
        * scale.float()[:, None, :, None]
    ).view(8192, 4096).to(torch.bfloat16)
    lo = args.rank * RANK_OUT
    return logical[lo : lo + RANK_OUT].contiguous().view(GROUPS, RANK_OUT, 4096)


def build_freqs(args: argparse.Namespace) -> torch.Tensor:
    from sglang.kernels.ops.attention.deepseek_v4_rope import precompute_freqs_cis

    config = json.loads((args.model_dir / "config.json").read_text())
    ratio = config["compress_ratios"][args.layer]
    if ratio not in (4, 128):
        raise ValueError(f"layer {args.layer} is not a compressed-RoPE layer: {ratio=}")
    scaling = config["rope_scaling"]
    return precompute_freqs_cis(
        dim=config["qk_rope_head_dim"],
        seqlen=config["max_position_embeddings"],
        original_seq_len=scaling["original_max_position_embeddings"],
        base=config["compress_rope_theta"],
        factor=scaling["factor"],
        beta_fast=scaling["beta_fast"],
        beta_slow=scaling["beta_slow"],
    )


def capture(fn):
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = fn()
    return graph, output


def graph_us(graph: torch.cuda.CUDAGraph, warmup: int, iters: int) -> float:
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


def error(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, float | bool]:
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
    prefix = args.dump_dir / f"layer_{args.layer}_rank_{args.rank}"
    raw_cpu = load_tensor(Path(f"{prefix}_attn_core.pt"))
    rotated_cpu = load_tensor(Path(f"{prefix}_attn_inverse_rope.pt"))
    positions_cpu = load_tensor(Path(f"{prefix}_positions.pt"))
    woa_dump_cpu = load_tensor(Path(f"{prefix}_wo_a.pt"))
    expected_shape = (M, HEADS, HEAD_DIM)
    if raw_cpu.shape != expected_shape or rotated_cpu.shape != expected_shape:
        raise ValueError(f"unexpected attention shapes {raw_cpu.shape} {rotated_cpu.shape}")
    if positions_cpu.shape != (M,):
        raise ValueError(f"unexpected positions {positions_cpu.shape}")

    raw = raw_cpu.cuda()
    rotated = rotated_cpu.cuda()
    positions = positions_cpu.cuda()
    freqs = build_freqs(args).cuda()
    weight = load_runtime_woa_shard(args).cuda()
    woa_dump = woa_dump_cpu.cuda()
    work_a = torch.empty_like(raw)
    work_g = torch.empty_like(raw)
    work_r = torch.empty_like(raw)
    work_c = torch.empty_like(raw)

    from sglang.kernels.ops.attention.deepseek_v4_rope import (
        apply_rotary_emb_triton,
        set_batched_rope,
    )

    set_batched_rope(True)

    def rope_inplace(x: torch.Tensor, pos: torch.Tensor = positions):
        apply_rotary_emb_triton(
            x[..., -ROPE_DIM:], freqs, positions=pos, inverse=True
        )

    def woa(x: torch.Tensor):
        return torch.einsum(
            "tgd,grd->tgr", x.view(M, GROUPS, 4096), weight
        )

    def path_a():
        work_a.copy_(raw)
        rope_inplace(work_a)
        return woa(work_a)

    def path_g():
        work_g.copy_(rotated)
        return woa(work_g)

    def path_r():
        work_r.copy_(raw)
        rope_inplace(work_r)
        return work_r

    def path_c():
        work_c.copy_(raw)
        return work_c

    # First-use compilation and exact reconstruction gates.
    eager_a = path_a()
    eager_g = path_g()
    torch.cuda.synchronize()
    print(
        {
            "dump_inverse_reconstruction": error(work_a, rotated),
            "woa_weight_reconstruction": error(eager_g, woa_dump),
            "A_vs_G": error(eager_a, eager_g),
        },
        flush=True,
    )

    # Exercise boundary positions using the same production inverse kernel.
    variant_positions = torch.tensor(
        [0, 4, 127, 128, 512] * 7,
        dtype=positions.dtype,
        device=positions.device,
    )[:M]
    variant_a = raw.clone()
    variant_b = raw.clone()
    rope_inplace(variant_a, variant_positions)
    rope_inplace(variant_b, variant_positions)
    print(
        {
            "variant_positions": [0, 4, 127, 128, 512],
            "variant_replay_exact": bool(torch.equal(variant_a, variant_b)),
            "variant_finite": bool(torch.isfinite(variant_a).all()),
        },
        flush=True,
    )

    graph_a, out_a = capture(path_a)
    graph_g, out_g = capture(path_g)
    graph_r, out_r = capture(path_r)
    graph_c, out_c = capture(path_c)
    for graph in (graph_a, graph_g, graph_r, graph_c):
        graph.replay()
    torch.cuda.synchronize()
    print(
        {
            "graph_A_vs_G": error(out_a, out_g),
            "graph_R_vs_dump": error(out_r, rotated),
            "graph_copy_exact": bool(torch.equal(out_c, raw)),
        },
        flush=True,
    )

    samples: dict[str, list[float]] = {key: [] for key in "AGRC"}
    # A/G/G/A is the decision-bearing ABBA. R/C are interleaved diagnostics.
    for _ in range(args.rounds):
        samples["A"].append(graph_us(graph_a, args.warmup, args.iters))
        samples["R"].append(graph_us(graph_r, args.warmup, args.iters))
        samples["G"].append(graph_us(graph_g, args.warmup, args.iters))
        samples["C"].append(graph_us(graph_c, args.warmup, args.iters))
        samples["G"].append(graph_us(graph_g, args.warmup, args.iters))
        samples["A"].append(graph_us(graph_a, args.warmup, args.iters))
    medians = {key: statistics.median(values) for key, values in samples.items()}
    exposed = medians["A"] - medians["G"]
    rope_net = medians["R"] - medians["C"]
    print(
        {
            "samples_us": samples,
            "medians_us": medians,
            "ideal_A_minus_G_us": exposed,
            "rope_R_minus_copy_us": rope_net,
            "ideal_speedup_percent": exposed / medians["A"] * 100.0,
            "continue_gate_us": 10.0,
            "passes_continue_gate": exposed >= 10.0,
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
