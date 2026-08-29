#!/usr/bin/env python3
"""Exact TP4 A4/R2 gate two-wave K-half oracle."""

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from scripts.rocm.bench_dsv4_tp4_m32_grouped_oracle import (
    _jit_gate_up_grouped_dpp,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8

E, M, T, I, H = 256, 32, 6, 512, 4096


@cache_once
def candidate_module():
    args = make_cpp_args(E, M, T, I, H, 4, 2, 8, 904)
    return load_jit(
        "gfx90a_fp4_gate_two_wave_khalf_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_gate_two_wave_khalf_oracle.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aFp4GateTwoWaveKHalfOracle<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def time_us(fn, warmup=20, iterations=100):
    for _ in range(warmup): fn()
    begin = torch.cuda.Event(enable_timing=True); end = torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(iterations): fn()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) * 1000.0 / iterations


def trimmed(values):
    values = sorted(values)
    return statistics.mean(values[1:-1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--recorder", default="/tmp/expert_distribution_recorder_1787803355.1855972.pt")
    p.add_argument("--mutations", type=int, default=100)
    p.add_argument("--rounds", type=int, default=7)
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    payload = torch.load(args.recorder, map_location="cpu", weights_only=False)
    ids = reconstruct_topk_from_counts(payload["logical_count"][37,34] // 8).cuda()
    metadata = make_metadata(ids, assignments=4)
    valid_blocks = (int(metadata.valid[0]) + 3) // 4
    if valid_blocks != 113:
        raise RuntimeError(f"barrier-safe oracle requires 113 blocks, got {valid_blocks}")
    torch.manual_seed(20260830)
    x = torch.randn((M,H), dtype=torch.bfloat16, device="cuda")
    xq, xs = per_token_group_quant_int8(x, 32)
    w = torch.randint(0,256,(E,2*I,H//2),dtype=torch.uint8,device="cuda")
    s = torch.full((E,2*I,H//32),127,dtype=torch.uint8,device="cuda")
    aout = torch.empty((M,T,I),dtype=torch.bfloat16,device="cuda")
    bout = torch.empty_like(aout)
    a = _jit_gate_up_grouped_dpp(E,M,T,I,H,4,2,8,2080,2)
    b = candidate_module()
    def run_a(): a.run(xq,xs,w,s,metadata.sorted_ids,metadata.sorted_experts,metadata.valid,aout,10.0)
    def run_b(): b.run(xq,xs,w,s,metadata.sorted_ids,metadata.sorted_experts,metadata.valid,bout,10.0)
    mutation = torch.empty_like(x)
    for index in range(args.mutations):
        mutation.normal_(); q,sc=per_token_group_quant_int8(mutation,32)
        xq.copy_(q); xs.copy_(sc); run_a(); run_b(); torch.cuda.synchronize()
        if not torch.equal(aout,bout):
            delta=(aout.float()-bout.float()).abs().max().item()
            raise RuntimeError(f"mutation={index} mismatch max_abs={delta}")
    print(f"CORRECTNESS mutations={args.mutations} intermediate_exact=True")
    values={"A":[],"B":[]}
    for _ in range(args.rounds):
        for name in ("A","B","B","A"):
            values[name].append(time_us(run_a if name=="A" else run_b,iterations=args.iterations))
    av,bv=trimmed(values["A"]),trimmed(values["B"])
    print(f"RESULT baseline_us={av:.3f} candidate_us={bv:.3f} saving_us={av-bv:.3f} gain_pct={(av/bv-1)*100:.3f}")


if __name__ == "__main__": main()
