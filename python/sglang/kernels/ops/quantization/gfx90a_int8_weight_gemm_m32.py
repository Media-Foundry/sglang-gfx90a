from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_SUPPORTED_SHAPES = {(32, 1536, 4096)}


def _config(a_tile: int) -> tuple[int, int, int]:
    # (output rows per wave, vector unroll, waves per workgroup).  A4 and A8
    # are both exposed solely for standalone tuning; neither is wired into a
    # model selector.
    return {4: (2, 1, 4), 8: (1, 1, 4)}[a_tile]


@cache_once
def _jit_gfx90a_int8_weight_gemm_m32_module(
    m: int, n: int, k: int, a_tile: int
) -> Module:
    out_rows, unroll, waves = _config(a_tile)
    args = make_cpp_args(m, n, k, a_tile, out_rows, unroll, waves)
    return load_jit(
        f"gfx90a_int8_weight_gemm_m32_a{a_tile}",
        *args,
        cuda_files=["gemm/gfx90a_int8_weight_gemm_m32.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aInt8WeightGemmM32Kernel<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_wave64_int8_weight_gemm_m32(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    a_tile: int = 8,
) -> torch.Tensor | None:
    """Experimental M32 projection; intentionally not used by production code."""
    shape = (x.shape[0], weight.shape[0], x.shape[1]) if x.ndim == 2 else None
    if (
        not torch.version.hip
        or shape not in _SUPPORTED_SHAPES
        or a_tile not in (4, 8)
        or weight.ndim != 2
        or weight.shape[1] != x.shape[1]
        or weight_scale.shape != (weight.shape[0],)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.int8
        or weight_scale.dtype != torch.float32
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not weight_scale.is_contiguous()
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    qx = torch.empty_like(x, dtype=torch.int8)
    x_scale = torch.empty((x.shape[0],), dtype=torch.float32, device=x.device)
    out = torch.empty(
        (x.shape[0], weight.shape[0]), dtype=torch.bfloat16, device=x.device
    )
    _jit_gfx90a_int8_weight_gemm_m32_module(*shape, a_tile).run(
        x, weight, weight_scale, qx, x_scale, out
    )
    return out
