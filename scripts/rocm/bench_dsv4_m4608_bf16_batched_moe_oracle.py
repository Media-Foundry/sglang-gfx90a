#!/usr/bin/env python3
"""Upper-bound oracle for transient BF16-expanded M4608 routed experts."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

E, PAD_M, H, I = 256, 128, 4096, 512


@cache_once
def dequant_module(n: int, k: int, blocks: int):
    args = make_cpp_args(E, n, k, blocks)
    return load_jit(
        "gfx90a_fp4_to_bf16_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_bf16_dequant_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ToBf16Oracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def elapsed_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def measure(name: str, n: int, k: int, args) -> None:
    packed = torch.randint(0, 256, (E, n, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.randint(118, 132, (E, n, k // 32), dtype=torch.uint8, device="cuda")
    weight = torch.empty((E, n, k), dtype=torch.bfloat16, device="cuda")
    x = torch.randn((E, PAD_M, k), dtype=torch.bfloat16, device="cuda")
    out = torch.empty((E, PAD_M, n), dtype=torch.bfloat16, device="cuda")
    module = dequant_module(n, k, args.blocks)

    def dequant():
        module.run(packed, scale, weight)

    def gemm():
        torch.bmm(x, weight.transpose(1, 2), out=out)

    def full():
        dequant()
        gemm()

    dequant()
    witness = weight.clone()
    dequant()
    torch.cuda.synchronize()
    if not torch.equal(weight, witness):
        raise RuntimeError(f"{name} dequant replay is not bitwise deterministic")

    samples = {"dequant": [], "gemm": [], "full": []}
    for _ in range(args.rounds):
        for key, fn in (("dequant", dequant), ("gemm", gemm), ("full", full)):
            samples[key].append(elapsed_us(fn, args.warmup, args.iterations))
    medians = {key: statistics.median(value) for key, value in samples.items()}
    print(f"RESULT stage={name} medians_us={medians} samples={samples}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=1664)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")
    torch.manual_seed(20260902)
    measure("gate_up", 2 * I, H, args)
    torch.cuda.empty_cache()
    measure("down", H, I, args)


if __name__ == "__main__":
    main()
