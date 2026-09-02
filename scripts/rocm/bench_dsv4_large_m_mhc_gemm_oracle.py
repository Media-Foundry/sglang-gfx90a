#!/usr/bin/env python3
"""ABBA large-M MHC pre-mix: decode wave64 versus BF16 GEMM pipeline."""

import argparse
import os
import statistics

import torch

from sglang.kernels.ops.layernorm.gfx90a_mhc_pre_mix import (
    _jit_gfx90a_mhc_pre_mix_module,
    gfx90a_mhc_pre_mix_wave64,
)


def time_ms(fn, iterations):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(True), torch.cuda.Event(True)
    begin.record()
    for _ in range(iterations):
        fn()
    end.record(); end.synchronize()
    return begin.elapsed_time(end) / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=2304)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--mutations", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(20260902)
    device = "cuda"
    x = torch.randn((args.m, 4, 4096), dtype=torch.bfloat16, device=device)
    fn = torch.randn((24, 16384), dtype=torch.float32, device=device) * 0.01
    fn_bf16 = fn.bfloat16().contiguous()
    raw = torch.empty((args.m, 24), dtype=torch.bfloat16, device=device)
    scale = torch.empty((args.m,), dtype=torch.float32, device=device)
    out = torch.empty((args.m, 1, 24), dtype=torch.float32, device=device)
    module = _jit_gfx90a_mhc_pre_mix_module()

    def wave():
        module.run(x, fn, out, 1e-6)
        return out

    def gemm():
        torch.mm(x.view(args.m, 16384), fn_bf16.t(), out=raw)
        module.rms_scale(x, scale, 1e-6)
        module.scale_mix(raw, scale, out)
        return out

    wave(); torch.cuda.synchronize(); reference = out.clone()
    gemm(); torch.cuda.synchronize(); candidate = out.clone()
    delta = (candidate - reference).abs()
    print(
        f"m={args.m} max_abs={delta.max().item():.6g} "
        f"mean_abs={delta.mean().item():.6g} "
        f"cos={torch.nn.functional.cosine_similarity(candidate.flatten(), reference.flatten(), dim=0).item():.9f}"
    )
    wave_samples, gemm_samples = [], []
    for _ in range(7):
        wave_samples.append(time_ms(wave, args.iterations))
        gemm_samples.append(time_ms(gemm, args.iterations))
        gemm_samples.append(time_ms(gemm, args.iterations))
        wave_samples.append(time_ms(wave, args.iterations))
    w = statistics.median(wave_samples)
    g = statistics.median(gemm_samples)
    print(f"wave_ms={w:.4f} gemm_ms={g:.4f} speedup={w/g:.3f}x")
    for mutation in range(args.mutations):
        x.normal_()
        fn.normal_(mean=0.0, std=0.01)
        os.environ.pop("SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM", None)
        reference = gfx90a_mhc_pre_mix_wave64(x, fn, 1e-6).clone()
        os.environ["SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM"] = "1"
        candidate = gfx90a_mhc_pre_mix_wave64(x, fn, 1e-6).clone()
        if not torch.isfinite(candidate).all():
            raise RuntimeError(f"mutation {mutation}: candidate is non-finite")
        cosine = torch.nn.functional.cosine_similarity(
            candidate.flatten(), reference.flatten(), dim=0
        ).item()
        if cosine < 0.99999:
            raise RuntimeError(f"mutation {mutation}: cosine={cosine}")
    if args.mutations:
        print(f"mutations={args.mutations} cosine_floor=0.99999 PASS")


if __name__ == "__main__":
    main()
