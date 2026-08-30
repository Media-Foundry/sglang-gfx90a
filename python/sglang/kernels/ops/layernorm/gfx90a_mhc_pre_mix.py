from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_mhc_pre_mix_module() -> Module:
    return load_jit(
        "gfx90a_mhc_pre_mix_wave64",
        cuda_files=["gemm/gfx90a_mhc_pre_mix.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aMhcPreMixKernel::run"),
            ("run_m64", "sglang::Gfx90aMhcPreMixKernel::run_m64"),
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
    if residual.shape[0] == 64:
        module.run_m64(residual, fn, mixes, rms_eps)
    else:
        module.run(residual, fn, mixes, rms_eps)
    return mixes
