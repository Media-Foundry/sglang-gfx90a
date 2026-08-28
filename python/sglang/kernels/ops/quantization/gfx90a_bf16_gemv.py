from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _config(n: int, k: int) -> tuple[int, int, int]:
    # MI250X decode shapes are sufficiently different that one launch geometry
    # leaves measurable bandwidth on the table. Values are (rows per wave,
    # vector unroll, waves per workgroup), tuned against the graph's three DSV4
    # projection shapes. Keep the conservative geometry for unknown shapes.
    if (n, k) == (2560, 160):
        return 2, 1, 8
    # Qwen4 full-attention PLE projection.  Eight waves each own three rows,
    # so the complete 24-row GEMV fits in one workgroup while sharing x once.
    if (n, k) == (24, 2560):
        return 3, 1, 8
    # TP4 Qwen4 LM-head shard. One row per wave avoids independent accumulator
    # pressure; a full 16-wave workgroup amortizes the shared-input staging.
    if (n, k) == (62080, 2560):
        return 1, 1, 16
    # Qwen compressed-QSA dense graph projects only the single 128-wide K
    # head.  One row per wave and four waves per workgroup minimizes the short
    # tail while exposing enough independent workgroups across the 128 rows.
    if (n, k) == (128, 2560):
        return 1, 1, 4
    rows, unroll, waves = {
        256: (1, 2, 8),
        8192: (2, 1, 4),
        4096: (1, 2, 4),
        1536: (1, 2, 8),
    }.get(n, (2, 2, 8))
    # Qwen4's hidden-width projections use K=2560/1536.  They cover complete
    # wave64 vector strides at unroll=1 (512 BF16 elements) but not unroll=2.
    if k % (64 * 8 * unroll) != 0:
        unroll = 1
    return rows, unroll, waves


@cache_once
def _jit_gfx90a_bf16_gemv_module(m: int, n: int, k: int) -> Module:
    rows, unroll, waves = _config(n, k)
    args = make_cpp_args(m, n, k, rows, unroll, waves)
    return load_jit(
        "gfx90a_bf16_gemv",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aBf16GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gfx90a_bf16_gate_up_swiglu_subgroup_module() -> Module:
    args = make_cpp_args(160, 2560, 8)
    return load_jit(
        "gfx90a_bf16_gate_up_swiglu_subgroup_oracle",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aBf16GateUpSwigluSubgroupKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gfx90a_bf16_fp32_gemv_module(n: int) -> Module:
    # Shape-specific CDNA2 wave64 geometry.  Keeping unroll=2 preserves the
    # existing FP32 accumulation order bit-for-bit while reducing excess
    # waves/row ownership for the small decode projections.
    rows, unroll, waves = {
        512: (1, 2, 8),
        1024: (1, 2, 4),
        2048: (1, 2, 4),
    }[n]
    args = make_cpp_args(n, 4096, rows, unroll, waves)
    return load_jit(
        "gfx90a_bf16_fp32_gemv",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aBf16Fp32GemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gfx90a_bf16_grouped_gemv_module(m: int) -> Module:
    args = make_cpp_args(m, 2, 1024, 4096, 1, 2, 4)
    return load_jit(
        "gfx90a_bf16_grouped_gemv",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aBf16GroupedGemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_wave64_bf16_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        # Independent wave64 rows beat rocBLAS for the latency tiers, but the
        # duplicated weight scan loses once M reaches the throughput tiers.
        or not (1 <= x.shape[0] <= 4)
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        # The Qwen4 PLE projection has 24 output rows.  The kernel already
        # guards tail rows, and eight-row alignment preserves vector output
        # tiers without excluding this latency-critical shape.
        or weight.shape[0] % 8 != 0
        or (weight.shape[1] % 512 != 0 and weight.shape[1] != 160)
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    out = torch.empty((x.shape[0], weight.shape[0]), dtype=x.dtype, device=x.device)
    _jit_gfx90a_bf16_gemv_module(
        x.shape[0], weight.shape[0], weight.shape[1]
    ).run(
        x, weight, out
    )
    return out


def gfx90a_bf16_gate_up_swiglu_subgroup(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.shape != (1, 2560)
        or weight.shape != (320, 2560)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None
    if out is None:
        out = torch.empty((1, 160), dtype=torch.bfloat16, device=x.device)
    if (
        out.shape != (1, 160)
        or out.dtype != torch.bfloat16
        or not out.is_contiguous()
        or out.device != x.device
    ):
        return None
    _jit_gfx90a_bf16_gate_up_swiglu_subgroup_module().run(x, weight, out)
    return out


def gfx90a_bf16_gate_up_swiglu_subgroup_oracle(
    x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor | None = None
) -> torch.Tensor:
    result = gfx90a_bf16_gate_up_swiglu_subgroup(x, weight, out)
    assert result is not None, "unsupported gfx90a shared gate/up oracle input"
    return result


def gfx90a_wave64_bf16_fp32_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.shape != (1, 4096)
        or weight.ndim != 2
        or weight.shape[0] not in (512, 1024, 2048)
        or weight.shape[1] != 4096
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None
    out = torch.empty((1, weight.shape[0]), dtype=torch.float32, device=x.device)
    _jit_gfx90a_bf16_fp32_gemv_module(weight.shape[0]).run(x, weight, out)
    return out


def gfx90a_wave64_bf16_grouped_gemv(
    x: torch.Tensor, weight: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 3
        # Grouped wo_a remains competitive through M=8; M=16 should reuse
        # weights through the batched einsum/GEMM path instead.
        or not (1 <= x.shape[0] <= 8)
        or x.shape[1:] != (2, 4096)
        or weight.shape != (2, 1024, 4096)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None
    out = torch.empty((x.shape[0], 2, 1024), dtype=torch.bfloat16, device=x.device)
    _jit_gfx90a_bf16_grouped_gemv_module(x.shape[0]).run(x, weight, out)
    return out
