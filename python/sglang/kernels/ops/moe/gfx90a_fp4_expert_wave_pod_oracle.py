"""Standalone JIT wrapper for the gfx90a same-expert wave-pod oracle.

This module is intentionally not imported by any production model or runner.
"""

from __future__ import annotations

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def jit_wave_pod_oracle(
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    assignments: int,
    rows: int,
    gate_blocks: int,
    down_blocks: int,
):
    if m not in (64, 96, 128):
        raise ValueError(f"wave-pod oracle only supports M64/M96/M128, got M={m}")
    if (e, t, i, k, assignments, rows) != (256, 6, 512, 4096, 4, 2):
        raise ValueError("wave-pod oracle requires TP4 DSV4 E256/T6/I512/K4096/A4/R2")
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, gate_blocks, down_blocks
    )
    return load_jit(
        "gfx90a_fp4_expert_wave_pod_oracle",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_wave_pod_oracle.cuh"],
        cuda_wrappers=[
            (
                "run_gate",
                f"sglang::Gfx90aFp4ExpertWavePodOracle<{args}>::run_gate",
            ),
            (
                "run_down",
                f"sglang::Gfx90aFp4ExpertWavePodOracle<{args}>::run_down",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )
