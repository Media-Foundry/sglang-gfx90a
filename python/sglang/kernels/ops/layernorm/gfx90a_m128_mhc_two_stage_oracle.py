"""Standalone-only JIT wrapper for the gfx90a M128 two-stage MHC oracle."""

from __future__ import annotations

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def jit_gfx90a_m128_mhc_two_stage_oracle(iters: int = 20):
    if iters not in (4, 8, 12, 20):
        raise ValueError(f"unsupported Sinkhorn iteration count: {iters}")
    return load_jit(
        f"gfx90a_m128_mhc_two_stage_oracle_i{iters}",
        cuda_files=["deepseek_v4/gfx90a_m128_mhc_two_stage_oracle.cuh"],
        cuda_wrappers=[
            ("producer", "sglang::Gfx90aM128MhcTwoStageOracle::producer"),
            ("consumer", "sglang::Gfx90aM128MhcTwoStageOracle::consumer"),
        ],
        extra_cuda_cflags=[
            "-O3",
            f"-DSGLANG_MHC_SINKHORN_ITERS={iters}",
            "-Rpass-analysis=kernel-resource-usage",
        ],
    )
