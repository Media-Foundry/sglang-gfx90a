#!/usr/bin/env python3
"""Compare one real DSV4 FP4 expert against a Torch dequantized oracle."""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from aiter.ops.shuffle import shuffle_scale_a16w4
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    gfx90a_fp4_expert_down,
    gfx90a_fp4_expert_gate_up,
)


FP4_TABLE = torch.tensor(
    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
    dtype=torch.float32,
)


def load_tensor(model: Path, weight_map: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(model / weight_map[name], framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def dequant_fp4(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    packed = weight.view(torch.uint8)
    table = FP4_TABLE.to(weight.device)
    values = torch.stack(
        (table[(packed & 0xF).long()], table[(packed >> 4).long()]), dim=-1
    ).flatten(-2)
    return values * scale.float().repeat_interleave(32, dim=-1)


def report(name: str, actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual_f = actual.float()
    expected_f = expected.float()
    diff = (actual_f - expected_f).abs()
    cosine = F.cosine_similarity(actual_f.flatten(), expected_f.flatten(), dim=0)
    print(
        f"{name}: max_abs={diff.max().item():.6g} "
        f"mean_abs={diff.mean().item():.6g} cosine={cosine.item():.9f} "
        f"neq_bf16={(actual.to(torch.bfloat16) != expected.to(torch.bfloat16)).sum().item()}"
    )
    print(f"  actual[:8]={actual_f.flatten()[:8].cpu().tolist()}")
    print(f"  expect[:8]={expected_f.flatten()[:8].cpu().tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("/home/pc/models/modelscope"))
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with open(args.model / "model.safetensors.index.json") as f:
        weight_map = json.load(f)["weight_map"]
    prefix = f"layers.{args.layer}.ffn.experts.{args.expert}"
    tensors = {
        part: load_tensor(args.model, weight_map, f"{prefix}.{part}").cuda()
        for part in (
            "w1.weight", "w1.scale", "w3.weight", "w3.scale",
            "w2.weight", "w2.scale",
        )
    }
    w13_i8 = torch.cat((tensors["w1.weight"], tensors["w3.weight"]), dim=0)
    s13_raw = torch.cat((tensors["w1.scale"], tensors["w3.scale"]), dim=0)
    w2_i8, s2_raw = tensors["w2.weight"], tensors["w2.scale"]

    # Match Fp8MoEMethod.process_weights_after_loading_block_quant exactly on gfx90a:
    # weights remain row-major while E8M0 scales use A16W4's shuffled layout.
    s13_ck = shuffle_scale_a16w4(s13_raw.reshape(-1, s13_raw.shape[-1]), 1, True)
    s2_ck = shuffle_scale_a16w4(s2_raw.reshape(-1, s2_raw.shape[-1]), 1, False)
    fp4_dtype = torch.float4_e2m1fn_x2
    w13_ck = w13_i8.unsqueeze(0).view(fp4_dtype)
    w2_ck = w2_i8.unsqueeze(0).view(fp4_dtype)

    torch.manual_seed(args.seed)
    x = (torch.randn((1, 4096), device="cuda", dtype=torch.bfloat16) / 20).contiguous()
    expert_ids = torch.zeros((1, 1), device="cuda", dtype=torch.int32)
    expert_mask = torch.ones((1,), device="cuda", dtype=torch.int32)
    live_count = torch.ones((1,), device="cuda", dtype=torch.int32)

    stage1_ck = gfx90a_fp4_expert_gate_up(
        x, w13_ck, s13_ck, expert_ids, expert_mask, live_count, 10.0
    )
    w13_ref = dequant_fp4(w13_i8, s13_raw)
    gate_up = x.float() @ w13_ref.T
    gate, up = gate_up.chunk(2, dim=-1)
    stage1_ref = (F.silu(gate.clamp(max=10.0)) * up.clamp(-10.0, 10.0)).to(
        torch.bfloat16
    ).view(1, 1, 2048)
    report("stage1", stage1_ck, stage1_ref)

    topk_weights = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    stage2_ck = gfx90a_fp4_expert_down(
        stage1_ref, w2_ck, s2_ck, expert_ids, expert_mask, topk_weights, live_count
    )
    w2_ref = dequant_fp4(w2_i8, s2_raw)
    stage2_ref = (stage1_ref.float().view(1, 2048) @ w2_ref.T).to(torch.bfloat16)
    report("stage2", stage2_ck, stage2_ref)


if __name__ == "__main__":
    main()
