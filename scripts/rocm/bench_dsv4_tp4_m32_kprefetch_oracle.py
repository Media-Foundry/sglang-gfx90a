#!/usr/bin/env python3
"""Reuse the strict grouped-gate oracle for the K-distance-1 prefetch kernel."""

from __future__ import annotations

from scripts.rocm import bench_dsv4_tp4_m32_paired_projection_oracle as harness
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_kprefetch(_unused_blocks: int):
    args = make_cpp_args(
        harness.E,
        harness.M,
        harness.T,
        harness.I,
        harness.H,
        harness.A,
        harness.R,
        harness.W,
        harness.G,
        harness.LUT,
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_up_kprefetch_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gate_up_kprefetch_oracle.cuh"
        ],
        cuda_wrappers=[(
            "run",
            f"sglang::Gfx90aFp4ExpertGateUpKPrefetchOracleKernel<{args}>::run",
        )],
        extra_cuda_cflags=["-O3"],
    )


if __name__ == "__main__":
    # The imported harness supplies mutation, graph replay and stage/full ABBA.
    # Its derived paired-block count is ignored by this fixed G2080 loader.
    harness._jit_paired = _jit_kprefetch
    harness.main()
