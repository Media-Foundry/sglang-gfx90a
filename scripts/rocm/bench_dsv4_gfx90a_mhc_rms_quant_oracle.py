#!/usr/bin/env python3
"""Benchmark the M32 MHC FFN RMSNorm + routed-gate INT8 producer oracle."""

from __future__ import annotations

import argparse
import statistics

import torch

from sglang.kernels.ops.layernorm.gfx90a_mhc_rms_quant_oracle import (
    gfx90a_mhc_rms_quant_oracle,
)
from sglang.kernels.ops.layernorm.mhc import (
    _gfx90a_mhc_rmsnorm_kernel,
    mhc_weighted_sum_triton,
)
from sglang.kernels.ops.quantization.gfx90a_int8_quant import (
    gfx90a_int8_group32_quant,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8


def timed(fn, warmup: int, iterations: int) -> float:
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
    return begin.elapsed_time(end) * 1000.0 / iterations


def abba(fn_a, fn_b, warmup: int, iterations: int, rounds: int):
    sa: list[float] = []
    sb: list[float] = []
    for _ in range(rounds):
        sa.append(timed(fn_a, warmup, iterations))
        sb.append(timed(fn_b, warmup, iterations))
        sb.append(timed(fn_b, warmup, iterations))
        sa.append(timed(fn_a, warmup, iterations))
    return sa, sb


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir", default="/tmp/dsv4_ffn_dump.f3ZQ89"
    )
    parser.add_argument("--layer", type=int, default=20)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--correctness-replays", type=int, default=100)
    args = parser.parse_args()

    if not torch.version.hip:
        raise RuntimeError("ROCm required")
    device = torch.device("cuda")
    prefix = f"{args.dump_dir}/layer_{args.layer}_rank_{args.rank}"
    residual = torch.load(
        f"{prefix}_ffn_mhc_residual.pt", map_location=device, weights_only=False
    ).contiguous()
    weight = torch.load(
        f"{prefix}_ffn_norm_weight.pt", map_location=device, weights_only=False
    ).contiguous()
    if residual.shape != (32, 4, 4096) or weight.shape != (4096,):
        raise RuntimeError(
            f"unexpected dump shapes residual={residual.shape} weight={weight.shape}"
        )
    torch.manual_seed(7)
    pre = torch.softmax(torch.randn((32, 4), device=device), dim=-1)
    x = mhc_weighted_sum_triton(residual, pre)
    if x is None:
        raise RuntimeError("weighted sum rejected real residual shape")
    out_a = torch.empty_like(x)
    out_b = torch.empty_like(x)
    q_b = torch.empty_like(x, dtype=torch.int8)
    s_b = torch.empty((32, 128), dtype=torch.float32, device=device)

    def rms(out: torch.Tensor) -> None:
        _gfx90a_mhc_rmsnorm_kernel[(32,)](
            x,
            weight,
            out,
            hidden_size=4096,
            eps=args.eps,
            BLOCK_H=4096,
            num_warps=8,
        )

    def run_a_triton():
        rms(out_a)
        return per_token_group_quant_int8(out_a, 32)

    def run_a_hip():
        rms(out_a)
        return gfx90a_int8_group32_quant(out_a)

    def run_b():
        gfx90a_mhc_rms_quant_oracle(x, weight, out_b, q_b, s_b, args.eps)
        return q_b, s_b

    q_a, s_a = run_a_triton()
    q_h, s_h = run_a_hip()
    run_b()
    torch.cuda.synchronize()
    exact = {
        "ffn_input": torch.equal(out_a, out_b),
        "q_vs_triton": torch.equal(q_a, q_b),
        "scale_vs_triton": torch.equal(s_a, s_b),
        "q_vs_hip": torch.equal(q_h, q_b),
        "scale_vs_hip": torch.equal(s_h, s_b),
    }
    print(
        f"correctness exact={exact} max_scale_triton={(s_a-s_b).abs().max().item():.8g} "
        f"max_scale_hip={(s_h-s_b).abs().max().item():.8g}",
        flush=True,
    )
    if not all(exact.values()):
        raise AssertionError("MHC RMS+quant oracle is not bitwise exact")

    for replay in range(args.correctness_replays):
        x.add_(torch.tensor((replay % 5) - 2, dtype=torch.bfloat16, device=device) / 64)
        q_a, s_a = run_a_triton()
        run_b()
        torch.cuda.synchronize()
        if not (
            torch.equal(out_a, out_b)
            and torch.equal(q_a, q_b)
            and torch.equal(s_a, s_b)
        ):
            raise AssertionError(f"mutated replay {replay} mismatch")
    print(
        f"correctness_mutations={args.correctness_replays} all_exact=True",
        flush=True,
    )

    triton_a, fused_b = abba(
        run_a_triton, run_b, args.warmup, args.iterations, args.rounds
    )
    hip_a, fused_b2 = abba(
        run_a_hip, run_b, args.warmup, args.iterations, args.rounds
    )
    tm, hm = statistics.median(triton_a), statistics.median(hip_a)
    bm = statistics.median(fused_b + fused_b2)
    print(
        f"ABBA triton_ref_us={tm:.3f} hip_ref_us={hm:.3f} fused_us={bm:.3f} "
        f"saved_vs_triton_us={tm-bm:.3f} delta_vs_triton_pct={(bm/tm-1)*100:+.2f} "
        f"saved_vs_hip_us={hm-bm:.3f} delta_vs_hip_pct={(bm/hm-1)*100:+.2f} "
        f"triton_samples={[round(v,3) for v in triton_a]} "
        f"hip_samples={[round(v,3) for v in hip_a]} "
        f"fused_samples={[round(v,3) for v in fused_b+fused_b2]}",
        flush=True,
    )


if __name__ == "__main__":
    main()
