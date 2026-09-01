#!/usr/bin/env python3
"""Full-stage upper bound for transient BF16-expanded DSV4 routed MoE."""

from __future__ import annotations

import argparse
import math
import statistics

import torch

from bench_dsv4_m4608_bf16_batched_moe_oracle import dequant_module, elapsed_us
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

E, T, H, I, ASSIGNMENTS = 256, 6, 4096, 512, 64


@cache_once
def helper_module(m: int, p: int, blocks: int):
    args = make_cpp_args(E, m, T, H, I, p, ASSIGNMENTS, blocks)
    return load_jit(
        "gfx90a_bf16_batched_moe_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_bf16_batched_moe_oracle.cuh"],
        cuda_wrappers=[
            ("pack", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::pack"),
            ("swiglu", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::swiglu"),
            ("reduce", f"sglang::Gfx90aBf16BatchedMoeOracle<{args}>::reduce"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def metadata(m: int, device):
    buckets: list[list[int]] = [[] for _ in range(E)]
    topk = torch.empty((m, T), dtype=torch.int32)
    expected_rows = torch.empty((m, T), dtype=torch.int32)
    for token in range(m):
        for slot in range(T):
            expert = (token * T + slot) % E
            topk[token, slot] = expert
            expected_rows[token, slot] = len(buckets[expert])
            buckets[expert].append((slot << 24) | token)
    sorted_ids: list[int] = []
    starts, counts = [], []
    for bucket in buckets:
        starts.append(len(sorted_ids) // ASSIGNMENTS)
        blocks = math.ceil(len(bucket) / ASSIGNMENTS)
        counts.append(blocks)
        sorted_ids.extend(bucket)
        sorted_ids.extend([m] * (blocks * ASSIGNMENTS - len(bucket)))
    p = max(counts) * ASSIGNMENTS
    return (
        p,
        topk.to(device),
        expected_rows.to(device),
        torch.tensor(sorted_ids, dtype=torch.int32, device=device),
        torch.tensor(starts, dtype=torch.int32, device=device),
        torch.tensor(counts, dtype=torch.int32, device=device),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, choices=(4608, 16384), default=4608)
    parser.add_argument("--padded-expert-rows", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=1664)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()
    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")
    device = torch.device("cuda")
    torch.manual_seed(20260902 + args.m)
    p, topk, expected_rows, sorted_ids, starts, counts = metadata(args.m, device)
    if args.padded_expert_rows:
        if args.padded_expert_rows < p:
            raise ValueError(
                f"padded-expert-rows {args.padded_expert_rows} is below required {p}"
            )
        p = args.padded_expert_rows
    route_rows = torch.full((args.m, T), -1, dtype=torch.int32, device=device)
    topk_weights = torch.rand((args.m, T), dtype=torch.float32, device=device)
    hidden = torch.randn((args.m, H), dtype=torch.bfloat16, device=device)
    expert_x = torch.empty((E, p, H), dtype=torch.bfloat16, device=device)
    gate_up = torch.empty((E, p, 2 * I), dtype=torch.bfloat16, device=device)
    intermediate = torch.empty((E, p, I), dtype=torch.bfloat16, device=device)
    expert_out = torch.empty((E, p, H), dtype=torch.bfloat16, device=device)
    output = torch.empty((args.m, H), dtype=torch.bfloat16, device=device)

    packed13 = torch.randint(
        0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device=device
    )
    scale13 = torch.randint(
        118, 132, (E, 2 * I, H // 32), dtype=torch.uint8, device=device
    )
    packed2 = torch.randint(
        0, 256, (E, H, I // 2), dtype=torch.uint8, device=device
    )
    scale2 = torch.randint(
        118, 132, (E, H, I // 32), dtype=torch.uint8, device=device
    )
    weight_workspace = torch.empty((E, 2 * I, H), dtype=torch.bfloat16, device=device)
    weight13 = weight_workspace
    weight2 = weight_workspace.view(-1)[: E * H * I].view(E, H, I)
    dequant13 = dequant_module(2 * I, H, args.blocks)
    dequant2 = dequant_module(H, I, args.blocks)
    helper = helper_module(args.m, p, args.blocks)

    def pack():
        helper.pack(hidden, sorted_ids, starts, counts, expert_x, route_rows)

    def gate_stage():
        dequant13.run(packed13, scale13, weight13)
        torch.bmm(expert_x, weight13.transpose(1, 2), out=gate_up)
        helper.swiglu(gate_up, intermediate, 10.0)

    def down_stage():
        dequant2.run(packed2, scale2, weight2)
        torch.bmm(intermediate, weight2.transpose(1, 2), out=expert_out)
        helper.reduce(expert_out, topk, route_rows, topk_weights, output)

    def full():
        pack()
        gate_stage()
        down_stage()

    pack()
    torch.cuda.synchronize()
    if not torch.equal(route_rows, expected_rows):
        diff = torch.count_nonzero(route_rows != expected_rows).item()
        raise RuntimeError(f"route-row mismatch count={diff}")
    full()
    witness = output.clone()
    full()
    torch.cuda.synchronize()
    if not torch.equal(output, witness):
        raise RuntimeError("full replay is not bitwise deterministic")

    samples = {"pack": [], "gate": [], "down": [], "full": []}
    for _ in range(args.rounds):
        for key, fn in (
            ("pack", pack),
            ("gate", gate_stage),
            ("down", down_stage),
            ("full", full),
        ):
            samples[key].append(elapsed_us(fn, args.warmup, args.iterations))
    medians = {key: statistics.median(values) for key, values in samples.items()}
    print(
        f"RESULT M={args.m} padded_expert_rows={p} "
        f"medians_us={medians} samples={samples}"
    )


if __name__ == "__main__":
    main()
