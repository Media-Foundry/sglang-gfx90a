#!/usr/bin/env python3
"""Extract runtime-equivalent layer-21 C128 projection weights without a service."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors import safe_open


def sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.contiguous().view(torch.uint8).numpy()).hexdigest()


def load(index: dict[str, str], model_dir: Path, key: str) -> torch.Tensor:
    with safe_open(model_dir / index[key], framework="pt", device="cpu") as handle:
        return handle.get_tensor(key).contiguous()


def dequant_block_fp8(weight: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if weight.ndim != 2 or weight.shape[0] % 128 or weight.shape[1] % 128:
        raise ValueError(f"unexpected block-FP8 weight shape {tuple(weight.shape)}")
    n_blocks, k_blocks = weight.shape[0] // 128, weight.shape[1] // 128
    if scale.shape != (n_blocks, k_blocks):
        raise ValueError(
            f"scale shape {tuple(scale.shape)} does not match {(n_blocks, k_blocks)}"
        )
    return (
        weight.float().view(n_blocks, 128, k_blocks, 128)
        * scale.float()[:, None, :, None]
    ).view(weight.shape).to(torch.bfloat16).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/home/pc/models/modelscope"))
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument(
        "--shape-input",
        type=Path,
        default=Path("/tmp/dsv4-layer20-m32/layer_20_attn_norm.pt"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    index = json.loads(
        (args.model_dir / "model.safetensors.index.json").read_text()
    )["weight_map"]
    prefix = f"layers.{args.layer}.attn"
    q_key, q_scale_key = f"{prefix}.wq_a.weight", f"{prefix}.wq_a.scale"
    kv_key, kv_scale_key = f"{prefix}.wkv.weight", f"{prefix}.wkv.scale"
    core_kv_key = f"{prefix}.compressor.wkv.weight"
    core_gate_key = f"{prefix}.compressor.wgate.weight"

    q = dequant_block_fp8(
        load(index, args.model_dir, q_key),
        load(index, args.model_dir, q_scale_key),
    )
    kv = dequant_block_fp8(
        load(index, args.model_dir, kv_key),
        load(index, args.model_dir, kv_scale_key),
    )
    core_kv = load(index, args.model_dir, core_kv_key)
    core_gate = load(index, args.model_dir, core_gate_key)
    wqkv = torch.cat((q, kv), dim=0).contiguous()
    core = torch.cat((core_kv, core_gate), dim=0).contiguous()
    x = torch.load(args.shape_input, map_location="cpu", weights_only=True).contiguous()

    expected = {
        "x": ((32, 4096), torch.bfloat16),
        "wqkv": ((1536, 4096), torch.bfloat16),
        "core": ((1024, 4096), torch.bfloat16),
    }
    for name, tensor in (("x", x), ("wqkv", wqkv), ("core", core)):
        shape, dtype = expected[name]
        if tuple(tensor.shape) != shape or tensor.dtype != dtype:
            raise ValueError(f"bad {name}: {tuple(tensor.shape)} {tensor.dtype}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x, args.output_dir / "shape_input_m32_bf16.pt")
    torch.save(wqkv, args.output_dir / "layer21_wqkv_a_bf16.pt")
    torch.save(core, args.output_dir / "layer21_core_wkv_gate_bf16.pt")
    manifest = {
        "layer": args.layer,
        "compress_ratio": 128,
        "input_source": str(args.shape_input),
        "input_is_shape_oracle_from_layer20": True,
        "checkpoint_keys": {
            "q": q_key,
            "q_scale": q_scale_key,
            "kv": kv_key,
            "kv_scale": kv_scale_key,
            "core_kv": core_kv_key,
            "core_gate": core_gate_key,
        },
        "tensors": {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": sha256(tensor),
            }
            for name, tensor in (("x", x), ("wqkv", wqkv), ("core", core))
        },
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
