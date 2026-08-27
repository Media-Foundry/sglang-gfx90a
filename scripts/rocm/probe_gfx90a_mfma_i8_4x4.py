#!/usr/bin/env python3
"""Dump raw lane mapping for v_mfma_i32_4x4x4i8."""

from __future__ import annotations

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import (
    gfx90a_mfma_i8_a4n4k32_oracle,
    gfx90a_mfma_i8_4x4_probe,
)


def pack_i8(values: list[int]) -> int:
    value = sum((item & 0xFF) << (8 * i) for i, item in enumerate(values))
    return value - (1 << 32) if value >= (1 << 31) else value


def main() -> None:
    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    device = torch.device("cuda")
    # Assume contiguous four-lane blocks initially.  Distinct row/column data
    # makes the raw dump sufficient to prove or reject that hypothesis.
    a = []
    b = []
    expected = []
    for block in range(16):
        matrix_a = [[1 + block + 3 * row + k for k in range(4)] for row in range(4)]
        matrix_b = [[2 - block + 2 * k - col for col in range(4)] for k in range(4)]
        expected.append(
            [
                [sum(matrix_a[row][k] * matrix_b[k][col] for k in range(4))
                 for col in range(4)]
                for row in range(4)
            ]
        )
        for lane_in_block in range(4):
            a.append(pack_i8(matrix_a[lane_in_block]))
            b.append(pack_i8([matrix_b[k][lane_in_block] for k in range(4)]))
    a_t = torch.tensor(a, dtype=torch.int32, device=device)
    b_t = torch.tensor(b, dtype=torch.int32, device=device)
    out = gfx90a_mfma_i8_4x4_probe(a_t, b_t).cpu()
    print("expected block0", expected[0])
    for lane in range(16):
        print(f"lane={lane:02d} block={lane//4} local={lane%4} out={out[lane].tolist()}")

    torch.manual_seed(17)
    x = torch.randint(-127, 128, (4, 32), dtype=torch.int8, device=device)
    weight = torch.randint(-12, 13, (4, 32), dtype=torch.int8, device=device)
    x_scale = torch.tensor([0.125, 0.25, 0.5, 1.0], device=device)
    weight_scale = torch.tensor([0.0625, 0.125, 0.25, 0.5], device=device)
    mfma, sdot, mfma_scaled, sdot_scaled = gfx90a_mfma_i8_a4n4k32_oracle(
        x, weight, x_scale, weight_scale
    )
    torch.cuda.synchronize()
    cpu_ref = x.cpu().to(torch.int32) @ weight.cpu().to(torch.int32).T
    print("mfma", mfma.cpu().tolist())
    print("sdot", sdot.cpu().tolist())
    print("cpu", cpu_ref.tolist())
    print(
        "integer_exact",
        torch.equal(mfma.cpu(), sdot.cpu()) and torch.equal(mfma.cpu(), cpu_ref),
        "scaled_exact",
        torch.equal(mfma_scaled, sdot_scaled),
    )


if __name__ == "__main__":
    main()
