#!/usr/bin/env python3
"""ABBA the affine and symmetric-decode MQ4 kernels on production M32 shapes."""

import os
import statistics
import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open

from sglang.kernels.ops.moe.gfx90a_mq4g128_moe import mq4g128_indexed
from sglang.srt.layers.quantization.mq4g128 import (
    _dequant_checkpoint_fp8,
    dequantize_mq4g128,
    fwht128,
    quantize_mq4g128,
)


def make_case(m: int, t: int, n: int, k: int, seed: int):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, k), dtype=torch.float32, device="cuda", generator=gen)
    symmetric_weight = torch.zeros(
        (128, n, k // 128, 68), dtype=torch.uint8, device="cuda"
    )
    symmetric_weight[..., 4:].random_(1, 16, generator=gen)
    scale = symmetric_weight[..., :4].view(torch.float32)
    scale.fill_(0.0125)
    affine_weight = torch.empty(
        (128, n, k // 128, 72), dtype=torch.uint8, device="cuda"
    )
    affine_weight[..., :4].copy_(symmetric_weight[..., :4])
    affine_weight[..., 4:8].view(torch.float32).copy_(-8.0 * scale)
    affine_weight[..., 8:].copy_(symmetric_weight[..., 4:])
    ids = torch.full((m, t), -1, dtype=torch.int32, device="cuda")
    ids.view(-1)[: min(m * t, 80)].random_(0, 128, generator=gen)
    return x, affine_weight, symmetric_weight, ids


def run(case, symmetric: bool):
    os.environ["SGLANG_QWEN4_GFX90A_MQ4G128_SYMMETRIC"] = (
        "1" if symmetric else "0"
    )
    x, affine_weight, symmetric_weight, ids = case
    return mq4g128_indexed(
        x,
        symmetric_weight if symmetric else affine_weight,
        ids,
        zero_invalid=True,
    )


def time_us(case, symmetric: bool, iters: int = 100) -> float:
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    run(case, symmetric)
    torch.cuda.synchronize()
    begin.record()
    for _ in range(iters):
        run(case, symmetric)
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iters


def checkpoint_error(model_path: str):
    model_dir = Path(model_path)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    keys = []
    for layer in (0, 24, 47):
        for projection in ("gate_proj", "down_proj"):
            keys.append(
                f"model.language_model.layers.{layer}.mlp.experts.0.{projection}.weight"
            )
    for key in keys:
        shard = model_dir / index["weight_map"][key]
        with safe_open(shard, framework="pt", device="cpu") as handle:
            weight = handle.get_tensor(key).cuda()
            scale = handle.get_tensor(f"{key}_scale_inv").cuda()
        reference = _dequant_checkpoint_fp8(weight.unsqueeze(0), scale.unsqueeze(0))[0]
        rotated = fwht128(reference)
        affine = dequantize_mq4g128(quantize_mq4g128(reference)).reshape_as(
            rotated
        )
        symmetric = dequantize_mq4g128(
            quantize_mq4g128(reference, symmetric=True)
        ).reshape_as(rotated)
        denom = torch.linalg.vector_norm(rotated)
        affine_rel = torch.linalg.vector_norm(rotated - affine) / denom
        symmetric_rel = torch.linalg.vector_norm(rotated - symmetric) / denom
        print(
            f"checkpoint {key}: affine_rel_l2={affine_rel.item():.8g} "
            f"symmetric_rel_l2={symmetric_rel.item():.8g} "
            f"error_ratio={(symmetric_rel / affine_rel).item():.5f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path")
    args = parser.parse_args()
    os.environ["SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M32"] = "1"
    for name, shape in (
        ("gate", (32, 10, 1280, 2560)),
        ("down", (320, 1, 2560, 640)),
    ):
        case = make_case(*shape, seed=29)
        reference = run(case, False)
        candidate = run(case, True)
        torch.cuda.synchronize()
        delta = reference - candidate
        rel = torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(reference)
        a, b = [], []
        for _ in range(15):
            a.append(time_us(case, False))
            b.append(time_us(case, True))
            b.append(time_us(case, True))
            a.append(time_us(case, False))
        print(
            f"{name}: affine_us={statistics.median(a):.3f} "
            f"symmetric_us={statistics.median(b):.3f} "
            f"speedup={statistics.median(a) / statistics.median(b):.4f} "
            f"max_abs={delta.abs().max().item():.8g} rel_l2={rel.item():.8g}"
        )
        baseline = candidate.clone()
        for _ in range(1000):
            replay = run(case, True)
        torch.cuda.synchronize()
        print(
            f"{name}: replay_1000_bitwise={torch.equal(baseline, replay)} "
            f"finite={torch.isfinite(replay).all().item()}"
        )
    if args.model_path:
        checkpoint_error(args.model_path)


if __name__ == "__main__":
    main()
