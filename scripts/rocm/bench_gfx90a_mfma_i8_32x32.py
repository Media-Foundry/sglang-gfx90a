#!/usr/bin/env python3
"""ABBA the raw CDNA2 M32N32K32 I8 MFMA against SDOT4."""

import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import _jit_module


def time_us(fn, iterations=10000):
    for _ in range(100):
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
    torch.manual_seed(31)
    x = torch.randint(-127, 128, (32, 32), dtype=torch.int8, device="cuda")
    w = torch.randint(-12, 13, (32, 32), dtype=torch.int8, device="cuda")
    mfma = torch.empty((32, 32), dtype=torch.int32, device="cuda")
    sdot = torch.empty_like(mfma)
    module = _jit_module()
    fm = lambda: module.m32n32k32_mfma(x, w, mfma, sdot)
    fs = lambda: module.m32n32k32_sdot(x, w, mfma, sdot)
    am, ass = [], []
    for _ in range(9):
        am.append(time_us(fm)); ass.append(time_us(fs))
        ass.append(time_us(fs)); am.append(time_us(fm))
    mm, sm = statistics.median(am), statistics.median(ass)
    print(f"mfma_us={mm:.4f} sdot_us={sm:.4f} speedup={sm/mm:.3f}x")


if __name__ == "__main__":
    main()
