#!/usr/bin/env python3
"""Correctness and ABBA for native gfx90a M16N16K16 I8 MFMA."""

import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import _jit_module


def time_us(fn, iterations=1000):
    for _ in range(50):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def main():
    torch.manual_seed(20260902)
    module = _jit_module()
    x = torch.randint(-127, 128, (16, 32), dtype=torch.int8, device="cuda")
    weight = torch.randint(-12, 13, (16, 32), dtype=torch.int8, device="cuda")
    mfma = torch.empty((16, 16), dtype=torch.int32, device="cuda")
    sdot = torch.empty_like(mfma)
    both = lambda: module.m16n16k32(x, weight, mfma, sdot)
    fm = lambda: module.m16n16k32_mfma(x, weight, mfma, sdot)
    fs = lambda: module.m16n16k32_sdot(x, weight, mfma, sdot)

    for replay in range(100):
        x.random_(-127, 128)
        weight.random_(-12, 13)
        both()
        torch.cuda.synchronize()
        if not torch.equal(mfma, sdot):
            delta = (mfma - sdot).abs().max().item()
            raise AssertionError(f"replay={replay} mismatch max_abs={delta}")

    mfma_samples, sdot_samples = [], []
    for _ in range(9):
        mfma_samples.append(time_us(fm))
        sdot_samples.append(time_us(fs))
        sdot_samples.append(time_us(fs))
        mfma_samples.append(time_us(fm))
    m = statistics.median(mfma_samples)
    s = statistics.median(sdot_samples)
    print("correctness=bitwise_exact mutations=100")
    print(f"mfma_us={m:.4f} sdot_us={s:.4f} speedup={s / m:.3f}x")


if __name__ == "__main__":
    main()
