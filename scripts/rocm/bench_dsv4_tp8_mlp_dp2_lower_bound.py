#!/usr/bin/env python3
"""Compute-only lower bound for TP8-attention + two TP4 MLP token groups.

The current path gives every GCD M=32 with an I=256 expert shard.  The hybrid
candidate would give each four-GCD group 16 rows and an I=512 shard; both
groups execute concurrently, so one GCD's M16/I512 stage is its compute
critical path.  This oracle intentionally excludes the required inter-group
row exchange, making a loss here a decisive rejection.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_down_grouped,
    _jit_gate_up_grouped,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


E, T, H, N = 256, 6, 4096, 4096
A, ROWS, WAVES, LDS = 4, 2, 8, 2


def metadata(topk: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    m = topk.shape[0]
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token, row in enumerate(topk.cpu().tolist()):
        for slot, expert in enumerate(row):
            buckets[expert].append((slot << 24) | token)
    ids: list[int] = []
    experts: list[int] = []
    for expert, bucket in enumerate(buckets):
        for offset in range(0, len(bucket), A):
            block = bucket[offset : offset + A]
            ids.extend(block + [m] * (A - len(block)))
            experts.append(expert)
    device = topk.device
    return (
        torch.tensor(ids, dtype=torch.int32, device=device),
        torch.tensor(experts, dtype=torch.int32, device=device),
        torch.tensor([len(ids), 0], dtype=torch.int32, device=device),
    )


class Stage:
    def __init__(self, m: int, i: int, topk: torch.Tensor) -> None:
        self.m, self.i = m, i
        self.ids, self.experts, self.valid = metadata(topk)
        self.x = torch.randn((m, H), dtype=torch.bfloat16, device="cuda")
        self.xq, self.xs = per_token_group_quant_int8(self.x, 32)
        self.w13 = torch.randint(
            0, 256, (E, 2 * i, H // 2), dtype=torch.uint8, device="cuda"
        )
        self.s13 = torch.full(
            (E, 2 * i, H // 32), 127, dtype=torch.uint8, device="cuda"
        )
        self.w2 = torch.randint(
            0, 256, (E, N, i // 2), dtype=torch.uint8, device="cuda"
        )
        self.s2 = torch.full(
            (E, N, i // 32), 127, dtype=torch.uint8, device="cuda"
        )
        self.weights = torch.rand((m, T), dtype=torch.float32, device="cuda")
        self.intermediate = torch.empty(
            (m, T, i), dtype=torch.bfloat16, device="cuda"
        )
        self.partial = torch.empty((m, T, N), dtype=torch.float32, device="cuda")
        self.output = torch.empty((m, N), dtype=torch.bfloat16, device="cuda")

    def run(self, gate_blocks: int, down_blocks: int) -> torch.Tensor:
        gate = _jit_gate_up_grouped(
            E, self.m, T, self.i, H, A, ROWS, WAVES, gate_blocks, LDS
        )
        gate.run(
            self.xq, self.xs, self.w13, self.s13, self.ids, self.experts,
            self.valid, self.intermediate, 10.0,
        )
        iq, isc = per_token_group_quant_int8(self.intermediate, 32)
        down = _jit_down_grouped(
            E, self.m, T, N, self.i, A, ROWS, WAVES, down_blocks, LDS
        )
        down.run_partial(
            iq, isc, self.w2, self.s2, self.ids, self.experts, self.valid,
            self.weights, self.partial,
        )
        down.reduce(self.partial, self.output)
        return self.output


def measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iters):
        fn()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000 / iters


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rounds", type=int, default=7)
    args = p.parse_args()
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":")[0] != "gfx90a":
        raise RuntimeError("requires gfx90a ROCm")
    torch.manual_seed(17)
    topk32 = torch.stack([torch.randperm(E, device="cuda")[:T] for _ in range(32)]).int()
    baseline = Stage(32, 256, topk32)
    candidate = Stage(16, 512, topk32[:16])
    profiles = ((416, 416), (624, 624), (832, 832), (1040, 832), (1040, 1040))
    baseline_profile = (832, 832)
    baseline.run(*baseline_profile); candidate.run(*profiles[0]); torch.cuda.synchronize()
    a_samples = [measure(lambda: baseline.run(*baseline_profile), args.warmup, args.iters) for _ in range(args.rounds)]
    a = statistics.median(a_samples)
    print(f"baseline M32_I256 profile={baseline_profile} median_us={a:.3f} samples={[round(x,3) for x in a_samples]}")
    for profile in profiles:
        samples = [measure(lambda p=profile: candidate.run(*p), args.warmup, args.iters) for _ in range(args.rounds)]
        b = statistics.median(samples)
        print(
            f"candidate M16_I512 profile={profile} median_us={b:.3f} "
            f"delta_us={b-a:+.3f} delta_pct={(b/a-1)*100:+.2f} "
            f"samples={[round(x,3) for x in samples]}"
        )


if __name__ == "__main__":
    main()
