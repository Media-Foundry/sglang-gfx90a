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
