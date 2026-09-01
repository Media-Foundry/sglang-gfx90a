#!/usr/bin/env python3
"""M4608 MFMA64 gate oracle with expert-persistent A64 block traversal."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_gate_up_mfma32,
)

E, M, T, I, K = 256, 4608, 6, 512, 4096
ASSIGNMENTS, SPLIT = 64, 4


def make_metadata(device: torch.device):
    buckets: list[list[int]] = [[] for _ in range(E)]
    for token in range(M):
        for slot in range(T):
            expert = (token * T + slot) % E
            buckets[expert].append((slot << 24) | token)

    sorted_ids: list[int] = []
    sorted_experts: list[int] = []
    active_experts: list[int] = []
    block_starts: list[int] = []
    block_counts: list[int] = []
    for expert, bucket in enumerate(buckets):
        start = len(sorted_experts)
        for offset in range(0, len(bucket), ASSIGNMENTS):
            block = bucket[offset : offset + ASSIGNMENTS]
            sorted_ids.extend(block)
            sorted_ids.extend([M] * (ASSIGNMENTS - len(block)))
            sorted_experts.append(expert)
        active_experts.append(expert)
        block_starts.append(start)
        block_counts.append(len(sorted_experts) - start)
    return tuple(
        torch.tensor(value, dtype=torch.int32, device=device)
        for value in (
            sorted_ids,
            sorted_experts,
            [len(sorted_ids), 0],
            active_experts,
            block_starts,
            block_counts,
            [len(active_experts)],
        )
    )


@cache_once
def candidate(blocks: int):
    args = make_cpp_args(E, M, T, I, K, blocks, SPLIT, ASSIGNMENTS)
    return load_jit(
        "gfx90a_fp4_mfma64_expert_persistent_gate_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_mfma64_expert_persistent_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentGateOracle<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def elapsed_us(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record()
    end.synchronize()
    return begin.elapsed_time(end) * 1000 / iterations


def trimmed(values: list[float]) -> float:
    return statistics.mean(sorted(values)[1:-1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=416)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--mutations", type=int, default=20)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm is required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a is required, got {arch}")

    device = torch.device("cuda")
    torch.manual_seed(20260902)
    (
        sorted_ids,
        sorted_experts,
        num_valid,
        active_experts,
        block_starts,
        block_counts,
        num_active,
    ) = make_metadata(device)
    xq = torch.randint(-8, 9, (M, K), dtype=torch.int8, device=device)
    xs = torch.rand((M, K // 32), dtype=torch.float32, device=device)
    weight = torch.randint(
        0, 256, (E, 2 * I, K // 2), dtype=torch.uint8, device=device
    )
    scales = torch.randint(
        118, 132, (E, 2 * I, K // 32), dtype=torch.uint8, device=device
    )
    reference = torch.empty((M, T, I), dtype=torch.bfloat16, device=device)
    output = torch.empty_like(reference)
    ref_module = _jit_gate_up_mfma32(
        E, M, T, I, K, args.blocks, SPLIT, 1, ASSIGNMENTS, False
    )
    cand_module = candidate(args.blocks)

    def run_reference():
        ref_module.run(
            xq,
            xs,
            weight,
            scales,
            sorted_ids,
            sorted_experts,
            num_valid,
            reference,
            10.0,
        )

    def run_candidate():
        cand_module.run(
            xq,
            xs,
            weight,
            scales,
            sorted_ids,
            active_experts,
            block_starts,
            block_counts,
            num_active,
            output,
            10.0,
        )

    for mutation in range(args.mutations):
        xq.random_(-8, 9)
        xs.uniform_(0.001, 0.1)
        run_reference()
        run_candidate()
        torch.cuda.synchronize()
        if not torch.equal(reference, output):
            diff = (reference.float() - output.float()).abs()
            raise RuntimeError(
                f"mutation={mutation} mismatch max_abs={diff.max().item()} "
                f"count={torch.count_nonzero(diff).item()}"
            )
    print(f"CORRECTNESS mutations={args.mutations} bitwise_exact=True")

    samples = {"A": [], "B": []}
    for _ in range(args.rounds):
        for name, fn in (
            ("A", run_reference),
            ("B", run_candidate),
            ("B", run_candidate),
            ("A", run_reference),
        ):
            samples[name].append(elapsed_us(fn, args.warmup, args.iterations))
    a = trimmed(samples["A"])
    b = trimmed(samples["B"])
    print(
        f"RESULT M={M} blocks={args.blocks} reference_us={a:.3f} "
        f"candidate_us={b:.3f} delta_us={b-a:.3f} speedup_pct={(a/b-1)*100:.3f}"
    )
    print(f"SAMPLES reference={samples['A']}")
    print(f"SAMPLES candidate={samples['B']}")


if __name__ == "__main__":
    main()
