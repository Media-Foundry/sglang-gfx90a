#!/usr/bin/env python3
"""Validate the CDNA2 I8 M32N32K32 MFMA lane mapping."""

import torch

from sglang.kernels.ops.moe.gfx90a_mfma_i8_4x4_oracle import (
    gfx90a_mfma_i8_m32n32k32_oracle,
)


def main() -> None:
    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")
    torch.manual_seed(29)
    device = torch.device("cuda")
    for replay in range(100):
        x = torch.randint(-127, 128, (32, 32), dtype=torch.int8, device=device)
        w = torch.randint(-12, 13, (32, 32), dtype=torch.int8, device=device)
        mfma, sdot = gfx90a_mfma_i8_m32n32k32_oracle(x, w)
        expected = (x.cpu().int() @ w.cpu().int().T).to(device)
        torch.cuda.synchronize()
        if not torch.equal(mfma, expected):
            raise AssertionError(
                f"MFMA mismatch replay={replay} max={(mfma - expected).abs().max().item()}"
            )
        if not torch.equal(sdot, expected):
            raise AssertionError(
                f"SDOT mismatch replay={replay} max={(sdot - expected).abs().max().item()}"
            )
    print("PASS: 100 mutations, MFMA and SDOT are bitwise exact")


if __name__ == "__main__":
    main()
