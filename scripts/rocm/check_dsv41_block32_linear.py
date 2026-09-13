#!/usr/bin/env python3
"""Check V4.1 dense FP8 math against explicit dequantization of real weights.

This is a component oracle, not proof of full-model correctness. No checkpoint
is modified and only the requested projection plus a few activations enter HBM.
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("/media/PM983/deepseek-v4.1-flash"))
    parser.add_argument("--key", default="layers.0.attn.wq_a")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]

    def read(key):
        with safe_open(args.model_dir / index[key], framework="pt", device="cpu") as f:
            return f.get_tensor(key)

    from sglang.srt.layers.quantization.fp8_utils import (
        dispatch_w8a8_block_fp8_linear,
        sglang_per_token_group_quant_fp8,
    )

    torch.manual_seed(341)
    w = read(args.key + ".weight").cuda()
    scales = read(args.key + ".scale").float().cuda()
    n, k = w.shape
    assert (n // scales.shape[0], k // scales.shape[1]) == (32, 32)
    weight_ref = w.float() * scales.repeat_interleave(32, 0).repeat_interleave(32, 1)
    linear = dispatch_w8a8_block_fp8_linear([32, 32], act_scale_ue8m0=True)
    report = {"key": args.key, "shape": [n, k], "device": torch.cuda.get_device_name(), "cases": []}
    for rows in (1, 3, 17, 32):
        x = torch.randn(rows, k, device="cuda", dtype=torch.bfloat16)
        q, s = sglang_per_token_group_quant_fp8(x, 32, scale_ue8m0=True)
        actual = linear(x, w, [32, 32], scales)
        quant_actual = linear(q, w, [32, 32], scales, input_scale=s)
        # Keep the backend's quantized operands, but independently expand both
        # scale grids and use FP32 arithmetic to check the GEMM addressing.
        x_ref = q.float() * s.float().repeat_interleave(32, 1)
        expected = x_ref @ weight_ref.t()
        err = actual.float() - expected
        relative_l2 = (err.norm() / expected.norm().clamp_min(1e-20)).item()
        finite = bool(torch.isfinite(actual).all() and torch.isfinite(expected).all())
        same_entry = torch.equal(actual, quant_actual)
        report["cases"].append({
            "rows": rows, "finite": finite, "prequant_equal": same_entry,
            "max_abs": err.abs().max().item(), "relative_l2": relative_l2,
            "passed": finite and same_entry and relative_l2 < 0.01,
        })
    report["passed"] = all(case["passed"] for case in report["cases"])
    with args.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
