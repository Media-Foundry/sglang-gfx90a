#!/usr/bin/env python3
"""Bit-exact and ABBA oracle for G2080 DPP gate readiness publication."""

import argparse
import statistics

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8
from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from scripts.rocm.bench_dsv4_tp4_m32_grouped_oracle import (
    _jit_gate_up_grouped_dpp,
)

E, M, T, I, H = 256, 32, 6, 512, 4096
A4, R2, W8, G, LUT = 4, 2, 8, 2080, 2


@cache_once
def candidate_module():
    args = make_cpp_args(E, M, T, I, H, A4, R2, W8, G, LUT)
    return load_jit(
        "gfx90a_fp4_gate_producer_release_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_gate_producer_release_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4GateProducerReleaseOracle<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def time_us(fn, warmup=20, iterations=100):
    for _ in range(warmup):
        fn()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        fn()
    end.record(); end.synchronize()
    return start.elapsed_time(end) * 1000.0 / iterations


def trimmed(values):
    values = sorted(values)
    return statistics.mean(values[1:-1]) if len(values) > 2 else statistics.mean(values)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--recorder", default="/tmp/expert_distribution_recorder_1787803355.1855972.pt")
    args = p.parse_args()

    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    counts = payload["logical_count"][37, 34] // 8
    metadata = make_metadata(reconstruct_topk_from_counts(counts).cuda(), assignments=A4)
    expert_blocks = metadata.sorted_experts.numel()
    torch.manual_seed(20260830)
    x = torch.randn((M, H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    w = torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8, device="cuda")
    s = torch.full((E, 2 * I, H // 32), 127, dtype=torch.uint8, device="cuda")
    ref_out = torch.empty((M, T, I), dtype=torch.bfloat16, device="cuda")
    cand_out = torch.empty_like(ref_out)
    counters = torch.zeros(expert_blocks, dtype=torch.int32, device="cuda")
    ready = torch.zeros_like(counters)
    ref = _jit_gate_up_grouped_dpp(E, M, T, I, H, A4, R2, W8, G, LUT)
    cand = candidate_module()

    def run_ref():
        ref.run(xq, xs, w, s, metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, ref_out, 10.0)

    def run_cand():
        cand.run(xq, xs, w, s, metadata.sorted_ids, metadata.sorted_experts,
                 metadata.valid, counters, ready, cand_out, 10.0)

    mutation = torch.empty_like(x)
    for index in range(args.mutations):
        mutation.normal_()
        q, scale = per_token_group_quant_int8(mutation, 32)
        xq.copy_(q); xs.copy_(scale)
        run_ref(); run_cand(); torch.cuda.synchronize()
        if not torch.equal(ref_out, cand_out):
            delta = (ref_out.float() - cand_out.float()).abs().max().item()
            raise RuntimeError(f"mutation={index} mismatch max_abs={delta}")
    expected = args.mutations
    valid_blocks = (max(int(metadata.valid[0].item()), 0) + A4 - 1) // A4
    if not bool(torch.all(counters[:valid_blocks] == expected * 32)):
        raise RuntimeError("counter mismatch")
    if not bool(torch.all(ready[:valid_blocks] == expected)):
        raise RuntimeError("ready epoch mismatch")
    print(f"CORRECTNESS mutations={args.mutations} intermediate_exact=True epochs_exact=True")

    ref_values, cand_values = [], []
    for _ in range(args.rounds):
        for name in ("ref", "cand", "cand", "ref"):
            (ref_values if name == "ref" else cand_values).append(
                time_us(run_ref if name == "ref" else run_cand,
                        iterations=args.iterations)
            )
    rv, cv = trimmed(ref_values), trimmed(cand_values)
    print(f"RESULT ref_us={rv:.3f} candidate_us={cv:.3f} delta_us={cv-rv:.3f} delta_pct={(cv/rv-1)*100:.3f}")
    print(f"DECISION pass_delta_lt_10us={cv-rv < 10.0}")


if __name__ == "__main__":
    main()
