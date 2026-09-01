#!/usr/bin/env python3
"""Validate device-side direct addressing of AIter A16W4 shuffled weights."""

import torch

from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

from sglang.kernels.ops.moe.gfx90a_fp4_preshuffle_probe import (
    gfx90a_fp4_preshuffle_probe,
)


def run_case(*, gate_up: bool, experts: int, rows: int, hidden: int) -> None:
    generator = torch.Generator(device="cpu").manual_seed(20260902 + gate_up)
    raw_w = torch.randint(
        0, 256, (experts, rows, hidden // 2), dtype=torch.uint8,
        generator=generator,
    )
    raw_s = torch.randint(
        0, 256, (experts * rows, hidden // 32), dtype=torch.uint8,
        generator=generator,
    )
    shuffled_w = shuffle_weight_a16w4(raw_w.cuda(), 16, gate_up)
    shuffled_s = shuffle_scale_a16w4(raw_s.cuda(), experts, gate_up)
    q = 4096
    query = torch.stack(
        (
            torch.randint(experts, (q,), generator=generator),
            torch.randint(rows, (q,), generator=generator),
            torch.randint(hidden // 2, (q,), generator=generator),
        ),
        dim=1,
    ).to(torch.int32)
    got_w, got_s = gfx90a_fp4_preshuffle_probe(
        shuffled_w, shuffled_s, query.cuda(), gate_up=gate_up
    )
    expected_w = raw_w[query[:, 0], query[:, 1], query[:, 2]]
    scale_group = query[:, 2] // 16
    expected_s = raw_s[query[:, 0] * rows + query[:, 1], scale_group]
    torch.cuda.synchronize()
    if not torch.equal(got_w.cpu(), expected_w):
        raise AssertionError("preshuffled weight address mismatch")
    if not torch.equal(got_s.cpu(), expected_s):
        raise AssertionError("preshuffled scale address mismatch")
    print(
        f"gate_up={gate_up} shape=({experts},{rows},{hidden // 2}) "
        f"queries={q} weight_exact=True scale_exact=True"
    )


def main() -> None:
    if torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0] != "gfx90a":
        raise RuntimeError("gfx90a required")
    run_case(gate_up=True, experts=2, rows=128, hidden=256)
    run_case(gate_up=False, experts=2, rows=64, hidden=256)


if __name__ == "__main__":
    main()
