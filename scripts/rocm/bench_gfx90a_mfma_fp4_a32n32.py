#!/usr/bin/env python3
"""Correctness and ABBA for a real raw-FP4 A32xN32xK4096 tile."""

import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import _jit_module


def time_us(fn, iterations=200):
    for _ in range(20): fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iterations): fn()
    b.record(); b.synchronize()
    return a.elapsed_time(b) * 1000 / iterations


def main():
    torch.manual_seed(37)
    module = _jit_module()
    xq = torch.randint(-127, 128, (32, 4096), dtype=torch.int8, device="cuda")
    xs = torch.rand((32, 128), dtype=torch.float32, device="cuda") * 0.05
    w = torch.randint(0, 256, (32, 2048), dtype=torch.uint8, device="cuda")
    ws = torch.randint(118, 132, (32, 128), dtype=torch.uint8, device="cuda")
    mo = torch.empty((32, 32), dtype=torch.float32, device="cuda")
    so = torch.empty_like(mo)
    both = lambda: module.fp4_a32n32k4096(xq, xs, w, ws, mo, so)
    fm = lambda: module.fp4_a32n32k4096_mfma(xq, xs, w, ws, mo, so)
    fs = lambda: module.fp4_a32n32k4096_sdot(xq, xs, w, ws, mo, so)
    max_abs = 0.0
    for replay in range(100):
        xq.random_(-127, 128); xs.uniform_(1e-4, 0.05)
        w.random_(0, 256); ws.random_(118, 132); both(); torch.cuda.synchronize()
        delta = (mo - so).abs().max().item(); max_abs = max(max_abs, delta)
        if not torch.equal(mo, so):
            raise AssertionError(f"replay={replay} not bitwise exact max_abs={delta}")
    mm, ss = [], []
    for _ in range(9):
        mm.append(time_us(fm)); ss.append(time_us(fs))
        ss.append(time_us(fs)); mm.append(time_us(fm))
    m, s = statistics.median(mm), statistics.median(ss)
    print(f"correctness=bitwise_exact mutations=100 max_abs={max_abs}")
    print(f"mfma_us={m:.3f} sdot_us={s:.3f} speedup={s/m:.3f}x")


if __name__ == "__main__": main()
