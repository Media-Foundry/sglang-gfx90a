#!/usr/bin/env python3
"""Correctness and ABBA for a raw-FP4 A16xN16xK4096 tile."""

import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import _jit_module


def time_us(fn, iterations=200):
    for _ in range(20):
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
    xq = torch.randint(-127, 128, (16, 4096), dtype=torch.int8, device="cuda")
    xs = torch.rand((16, 128), dtype=torch.float32, device="cuda") * 0.05
    weight = torch.randint(0, 256, (16, 2048), dtype=torch.uint8, device="cuda")
    ws = torch.randint(118, 132, (16, 128), dtype=torch.uint8, device="cuda")
    mfma = torch.empty((16, 16), dtype=torch.float32, device="cuda")
    sdot = torch.empty_like(mfma)
    both = lambda: module.fp4_a16n16k4096(xq, xs, weight, ws, mfma, sdot)
    fm = lambda: module.fp4_a16n16k4096_mfma(xq, xs, weight, ws, mfma, sdot)
    fs = lambda: module.fp4_a16n16k4096_sdot(xq, xs, weight, ws, mfma, sdot)

    for replay in range(100):
        xq.random_(-127, 128)
        xs.uniform_(1e-4, 0.05)
        weight.random_(0, 256)
        ws.random_(118, 132)
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
    print(f"mfma_us={m:.3f} sdot_us={s:.3f} speedup={s / m:.3f}x")


if __name__ == "__main__":
    main()
