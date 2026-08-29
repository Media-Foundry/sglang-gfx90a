#!/usr/bin/env python3
"""Bit-exact and ABBA oracle for expert-owned W8 DPP gate publication."""

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
def expert_owned_module(owner_ctas):
    args = make_cpp_args(E, M, T, I, H, A4, R2, W8, owner_ctas, LUT)
    return load_jit(
        "gfx90a_fp4_gate_expert_owned_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_gate_expert_owned_oracle.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4GateExpertOwnedOracle<{args}>::run")
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
    ref = _jit_gate_up_grouped_dpp(E, M, T, I, H, A4, R2, W8, G, LUT)
    owner_ctas_values = (1, 4, 8)
    states = {
        ctas: {
            "out": torch.empty_like(ref_out),
            "counter": torch.zeros(expert_blocks, dtype=torch.int32, device="cuda"),
            "ready": torch.zeros(expert_blocks, dtype=torch.int32, device="cuda"),
            "module": expert_owned_module(ctas),
        }
        for ctas in owner_ctas_values
    }

    def run_ref():
        ref.run(xq, xs, w, s, metadata.sorted_ids, metadata.sorted_experts,
                metadata.valid, ref_out, 10.0)

    def run_cand(ctas):
        state = states[ctas]
        state["module"].run(
            xq, xs, w, s, metadata.sorted_ids, metadata.sorted_experts,
            metadata.valid, state["counter"], state["ready"], state["out"], 10.0)

    mutation = torch.empty_like(x)
    for index in range(args.mutations):
        mutation.normal_()
        q, scale = per_token_group_quant_int8(mutation, 32)
        xq.copy_(q); xs.copy_(scale)
        run_ref()
        for ctas in owner_ctas_values:
            run_cand(ctas)
        torch.cuda.synchronize()
        for ctas in owner_ctas_values:
            if not torch.equal(ref_out, states[ctas]["out"]):
                delta = (ref_out.float() - states[ctas]["out"].float()).abs().max().item()
                raise RuntimeError(f"mutation={index} ctas={ctas} mismatch max_abs={delta}")
    expected = args.mutations
    valid_blocks = (max(int(metadata.valid[0].item()), 0) + A4 - 1) // A4
    for ctas in owner_ctas_values:
        if not bool(torch.all(states[ctas]["counter"][:valid_blocks] == expected * ctas)):
            raise RuntimeError(f"counter mismatch ctas={ctas}")
        if not bool(torch.all(states[ctas]["ready"][:valid_blocks] == expected)):
            raise RuntimeError(f"ready epoch mismatch ctas={ctas}")
    print(f"CORRECTNESS mutations={args.mutations} profiles={owner_ctas_values} intermediate_exact=True epochs_exact=True")

    values = {"ref": []} | {f"owner{ctas}": [] for ctas in owner_ctas_values}
    for _ in range(args.rounds):
        order = ["ref"] + [f"owner{x}" for x in owner_ctas_values]
        for name in order + list(reversed(order)):
            fn = run_ref if name == "ref" else (lambda c=int(name[5:]): run_cand(c))
            values[name].append(time_us(fn, iterations=args.iterations))
    summary = {name: trimmed(sample) for name, sample in values.items()}
    rv = summary["ref"]
    for ctas in owner_ctas_values:
        cv = summary[f"owner{ctas}"]
        print(f"RESULT owner_ctas={ctas} ref_us={rv:.3f} candidate_us={cv:.3f} delta_us={cv-rv:.3f} delta_pct={(cv/rv-1)*100:.3f}")


if __name__ == "__main__":
    main()
