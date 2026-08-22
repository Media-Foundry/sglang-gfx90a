from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


def _config(n: int, k: int) -> tuple[int, int, int] | None:
    return {
        (8192, 1024): (4, 1, 4),
        (4096, 2048): (2, 2, 4),
        (1536, 4096): (2, 1, 4),
    }.get((n, k))


@cache_once
def _jit_gfx90a_int8_weight_gemv_module(n: int, k: int) -> Module:
    config = _config(n, k)
    assert config is not None
    rows, unroll, waves = config
    args = make_cpp_args(n, k, rows, unroll, waves)
    return load_jit(
        "gfx90a_int8_weight_gemv",
        *args,
        cuda_files=["gemm/gfx90a_int8_weight_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aInt8WeightGemvKernel<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_wave64_int8_weight_gemv(
    x: torch.Tensor, weight: torch.Tensor, weight_scale: torch.Tensor
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or x.shape[0] != 1
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or weight_scale.shape != (weight.shape[0],)
        or _config(weight.shape[0], weight.shape[1]) is None
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

    out = torch.empty((1, weight.shape[0]), dtype=x.dtype, device=x.device)
    _jit_gfx90a_int8_weight_gemv_module(weight.shape[0], weight.shape[1]).run(
        x, weight, weight_scale, out
    )
    return out
