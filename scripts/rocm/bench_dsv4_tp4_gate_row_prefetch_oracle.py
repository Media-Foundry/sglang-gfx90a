#!/usr/bin/env python3
"""Strict TP4 A4/R2 grouped-gate same-group row-prefetch oracle."""

from scripts.rocm import bench_dsv4_tp4_m32_paired_projection_oracle as harness
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_row_prefetch(_unused_blocks: int):
    args = make_cpp_args(
        harness.E, harness.M, harness.T, harness.I, harness.H,
        harness.A, harness.R, harness.W, harness.G, harness.LUT,
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_row_prefetch_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh"],
        cuda_wrappers=[(
            "run",
            f"sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{args}>::run",
        )],
        extra_cuda_cflags=["-O3", "-save-temps"],
    )


if __name__ == "__main__":
    harness._jit_paired = _jit_row_prefetch
    harness.main()
