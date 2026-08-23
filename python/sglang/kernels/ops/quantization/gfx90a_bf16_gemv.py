from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _config(n: int) -> tuple[int, int, int]:
    # MI250X decode shapes are sufficiently different that one launch geometry
    # leaves measurable bandwidth on the table. Values are (rows per wave,
    # vector unroll, waves per workgroup), tuned against the graph's three DSV4
    # projection shapes. Keep the conservative geometry for unknown shapes.
    return {
        256: (1, 2, 8),
        8192: (2, 1, 4),
        4096: (1, 2, 4),
        1536: (1, 2, 8),
    }.get(n, (2, 2, 8))


@cache_once
def _jit_gfx90a_bf16_gemv_module(n: int, k: int) -> Module:
    rows, unroll, waves = _config(n)
    args = make_cpp_args(n, k, rows, unroll, waves)
    return load_jit(
        "gfx90a_bf16_gemv",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aBf16GemvKernel<{args}>::run")],
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
def _jit_gfx90a_bf16_grouped_gemv_module() -> Module:
    args = make_cpp_args(2, 1024, 4096, 1, 2, 4)
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
        or x.shape[0] != 1
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or weight.shape[0] % 16 != 0
        or weight.shape[1] % 1024 != 0
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _jit_gfx90a_bf16_gemv_module(weight.shape[0], weight.shape[1]).run(
        x, weight, out
    )
    return out


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
        or x.shape != (1, 2, 4096)
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
    out = torch.empty((1, 2, 1024), dtype=torch.bfloat16, device=x.device)
    _jit_gfx90a_bf16_grouped_gemv_module().run(x, weight, out)
    return out
