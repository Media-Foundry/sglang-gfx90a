#!/usr/bin/env python3
"""Compare captured routed partials with raw-checkpoint W4A16 CPU math.

This intentionally omits online A4 activation quantization: use it to identify
gross layout/scale errors, NOT as a bit-exact MXFP4 acceptance test.
Only the last captured token's selected experts are loaded, one shard at a time.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--model-dir", type=Path, default=Path("/media/PM983/deepseek-v4.1-flash"))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--layer", type=int, default=0)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(4)
    index = json.loads((args.model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.])

    def weight(key, rank, projection):
        with safe_open(args.model_dir / index[key + ".weight"], framework="pt", device="cpu") as f:
            packed = f.get_tensor(key + ".weight").view(torch.uint8)
        with safe_open(args.model_dir / index[key + ".scale"], framework="pt", device="cpu") as f:
            scale = f.get_tensor(key + ".scale").float()
        if projection != "w2":
            assert packed.shape[0] % args.tp == 0
            width = packed.shape[0] // args.tp
            packed = packed[rank * width:(rank + 1) * width]
            scale = scale[rank * width:(rank + 1) * width]
        else:
            assert packed.shape[1] % args.tp == 0 and scale.shape[1] % args.tp == 0
            width = packed.shape[1] // args.tp
            packed = packed[:, rank * width:(rank + 1) * width]
            groups = scale.shape[1] // args.tp
            scale = scale[:, rank * groups:(rank + 1) * groups]
        codes = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
        assert codes.shape[1] == scale.shape[1] * 32
        return lut[codes] * scale.repeat_interleave(32, -1)

    result = {"oracle": "raw FP4 weights, BF16 activation (not online A4)", "ranks": []}
    for rank in range(args.tp):
        paths = list(args.trace.glob(f"rank{rank}-firstdiv-FusedMoE-*-call0-*.pt"))
        assert len(paths) == 1, paths
        trace = torch.load(paths[0], weights_only=True)
        x = trace["args"][0][-1:].float()
        top = trace["args"][1]
        route_weights, experts = top[0][-1], top[1][-1]
        ref = torch.zeros_like(x)
        for eid, routing in zip(experts.tolist(), route_weights.tolist()):
            key = f"layers.{args.layer}.ffn.experts.{eid}"
            gate = F.linear(x, weight(key + ".w1", rank, "w1")).bfloat16().float().clamp(max=10)
            up = F.linear(x, weight(key + ".w3", rank, "w3")).bfloat16().float().clamp(-10, 10)
            hidden = (F.silu(gate) * up * routing).bfloat16().float()
            ref += F.linear(hidden, weight(key + ".w2", rank, "w2")).bfloat16().float()
        actual = trace["output"][-1:].float()
        row = {"rank": rank, "expert_ids": experts.tolist(),
               "actual_rms": actual.square().mean().sqrt().item(),
               "reference_rms": ref.square().mean().sqrt().item(),
               "relative_l2": ((actual - ref).norm() / ref.norm().clamp_min(1e-20)).item(),
               "cosine": F.cosine_similarity(actual, ref).item()}
        result["ranks"].append(row)
        print(json.dumps(row), flush=True)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
