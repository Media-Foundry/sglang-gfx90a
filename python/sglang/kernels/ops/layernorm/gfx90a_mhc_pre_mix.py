from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_large_m_bf16_fn_cache: dict[tuple[int, int, int], torch.Tensor] = {}


@cache_once
def _jit_gfx90a_mhc_pre_mix_module() -> Module:
    return load_jit(
        "gfx90a_mhc_pre_mix_wave64",
        cuda_files=["gemm/gfx90a_mhc_pre_mix.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aMhcPreMixKernel::run"),
            ("run_m64", "sglang::Gfx90aMhcPreMixKernel::run_m64"),
            ("rms_scale", "sglang::Gfx90aMhcPreMixKernel::rms_scale"),
            ("scale_mix", "sglang::Gfx90aMhcPreMixKernel::scale_mix"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def preload_gfx90a_mhc_pre_mix_wave64() -> None:
    """Load/compile the module before distributed CUDA graph capture."""
    _jit_gfx90a_mhc_pre_mix_module()


def gfx90a_mhc_pre_mix_wave64(
    residual: torch.Tensor, fn: torch.Tensor, rms_eps: float
) -> torch.Tensor | None:
    """Native HIP wave64 MHC pre-mix for DSV4 decode on gfx90a."""
    if (
        not torch.version.hip
        or residual.ndim != 3
        or residual.shape[0] < 1
        or residual.shape[1:] != (4, 4096)
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or fn.shape != (24, 16384)
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    mixes = torch.empty(
        (residual.shape[0], 1, 24), dtype=torch.float32, device=residual.device
    )
    module = _jit_gfx90a_mhc_pre_mix_module()
    if (
        residual.shape[0] >= 2048
        and os.getenv("SGLANG_DSV4_GFX90A_MHC_LARGE_M_BF16_GEMM", "0") == "1"
        and not torch.cuda.is_current_stream_capturing()
    ):
        device_index = residual.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        key = (device_index, fn.data_ptr(), fn._version)
        fn_bf16 = _large_m_bf16_fn_cache.get(key)
        if fn_bf16 is None:
            fn_bf16 = fn.to(dtype=torch.bfloat16).contiguous()
            _large_m_bf16_fn_cache[key] = fn_bf16
        raw = torch.empty(
            (residual.shape[0], 24),
            dtype=torch.bfloat16,
            device=residual.device,
        )
        scale = torch.empty(
            (residual.shape[0],), dtype=torch.float32, device=residual.device
        )
        torch.mm(residual.view(residual.shape[0], 16384), fn_bf16.t(), out=raw)
        module.rms_scale(residual, scale, rms_eps)
        module.scale_mix(raw, scale, mixes)
        return mixes
    if residual.shape[0] == 64:
        module.run_m64(residual, fn, mixes, rms_eps)
    else:
        module.run(residual, fn, mixes, rms_eps)
    return mixes
