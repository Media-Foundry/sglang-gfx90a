#!/usr/bin/env python3
"""Check production AIter scale shuffle -> CK logical scale conversion."""

import torch
from aiter.ops.shuffle import shuffle_scale_a16w4
from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import _logical_a16w4_scales


def main():
    torch.manual_seed(2709)
    for intermediate in (256, 512):
        for gate_up, rows, groups in (
            (True, 2 * intermediate, 128), (False, 4096, intermediate // 32)
        ):
            for _ in range(10):
                # CPU only: byte layout, including all possible E8M0 encodings.
                raw = torch.randint(0, 256, (3, rows, groups), dtype=torch.uint8)
                shuffled = shuffle_scale_a16w4(raw.view(-1, groups), 3, gate_up)
                restored = _logical_a16w4_scales(
                    shuffled, 3, rows, groups, gate_up=gate_up
                )
                assert torch.equal(raw, restored)
            print(f"I={intermediate} gate_up={gate_up}: 10 mutations byte-exact")


if __name__ == "__main__":
    main()
