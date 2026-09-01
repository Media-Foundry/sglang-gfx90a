#!/usr/bin/env python3
"""M4608 MFMA64 down oracle with expert-persistent A64 traversal."""

from __future__ import annotations

import argparse

import torch

from bench_dsv4_gfx90a_mfma64_expert_persistent_gate import (
    ASSIGNMENTS,
    E,
    I,
    M,
    T,
    elapsed_us,
    make_metadata,
    trimmed,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_mfma32

N, K, SPLIT = 4096, I, 2


@cache_once
def candidate(blocks: int):
    args = make_cpp_args(E, M, T, N, K, blocks, SPLIT, ASSIGNMENTS)
    return load_jit(
        "gfx90a_fp4_mfma64_expert_persistent_down_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_mfma64_expert_persistent_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentDownOracle<{args}>::run_partial",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blocks", type=int, default=624)
    parser.add_argument("--reference-blocks", type=int, default=624)
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
        _sorted_experts,
        num_valid,
        active_experts,
        block_starts,
        block_counts,
        num_active,
    ) = make_metadata(device)
    sorted_experts = _sorted_experts
    xq = torch.randint(-8, 9, (M, T, K), dtype=torch.int8, device=device)
    xs = torch.rand((M, T, K // 32), dtype=torch.float32, device=device)
    weight = torch.randint(
        0, 256, (E, N, K // 2), dtype=torch.uint8, device=device
    )
    scales = torch.randint(
        118, 132, (E, N, K // 32), dtype=torch.uint8, device=device
    )
    topk_weights = torch.rand((M, T), dtype=torch.float32, device=device)
    reference = torch.empty((M, T, N), dtype=torch.float32, device=device)
    output = torch.empty_like(reference)
    # The production wrapper also launches fixed-slot reduction.  Call the
    # underlying kernel through a tiny oracle wrapper so producer timing and
    # partial correctness remain isolated.
    ref_module = _jit_down_mfma32(
        E, M, T, N, K, args.reference_blocks, SPLIT, 1, ASSIGNMENTS, False
    )
    cand_module = candidate(args.blocks)
    reduced_dummy = torch.empty((M, N), dtype=torch.bfloat16, device=device)

    def run_reference():
        ref_module.run(
            xq,
            xs,
            weight,
            scales,
            sorted_ids,
            sorted_experts,
            num_valid,
            topk_weights,
            reference,
            reduced_dummy,
        )

    def run_candidate():
        cand_module.run_partial(
            xq,
            xs,
            weight,
            scales,
            sorted_ids,
            active_experts,
            block_starts,
            block_counts,
            num_active,
            topk_weights,
            output,
        )

    # Reference `run` includes reduction, but its partial is final before the
    # reduction launch. Candidate writes the same fixed slots.
    for mutation in range(args.mutations):
        xq.random_(-8, 9)
        xs.uniform_(0.001, 0.1)
        topk_weights.uniform_()
        run_reference()
        run_candidate()
        torch.cuda.synchronize()
        if not torch.equal(reference, output):
            diff = (reference - output).abs()
            raise RuntimeError(
                f"mutation={mutation} mismatch max_abs={diff.max().item()} "
                f"count={torch.count_nonzero(diff).item()}"
            )
    print(f"CORRECTNESS mutations={args.mutations} partial_bitwise_exact=True")

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
        f"RESULT M={M} reference_blocks={args.reference_blocks} "
        f"candidate_blocks={args.blocks} reference_with_reduce_us={a:.3f} "
        f"candidate_partial_us={b:.3f} delta_us={b-a:.3f} "
        f"speedup_pct={(a/b-1)*100:.3f}"
    )
    print(f"SAMPLES reference={samples['A']}")
    print(f"SAMPLES candidate={samples['B']}")


if __name__ == "__main__":
    main()
