from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_mhc_sinkhorn_module(iters: int) -> Module:
    return load_jit(
        f"gfx90a_mhc_sinkhorn_wave64_i{iters}",
        cuda_files=["deepseek_v4/gfx90a_mhc_sinkhorn.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aMhcSinkhornKernel::run")],
        extra_cuda_cflags=["-O3", f"-DSGLANG_MHC_SINKHORN_ITERS={iters}"],
    )


def preload_gfx90a_mhc_sinkhorn(iters: int) -> None:
    if iters not in (4, 8, 12, 20):
        raise ValueError(f"unsupported gfx90a MHC Sinkhorn iterations: {iters}")
    _jit_gfx90a_mhc_sinkhorn_module(iters)


def gfx90a_mhc_sinkhorn_wave64(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    eps: float,
    iters: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not torch.version.hip
        or mixes.ndim != 3
        or mixes.shape[0] < 1
        or mixes.shape[1:] != (1, 24)
        or mixes.dtype != torch.float32
        or not mixes.is_contiguous()
        or hc_scale.shape != (3,)
        or hc_scale.dtype != torch.float32
        or not hc_scale.is_contiguous()
        or hc_base.shape != (24,)
        or hc_base.dtype != torch.float32
        or not hc_base.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(mixes.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    tokens = mixes.shape[0]
    pre = torch.empty((tokens, 1, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty_like(pre)
    comb = torch.empty(
        (tokens, 1, 4, 4), dtype=torch.float32, device=mixes.device
    )
    if iters is None:
        iters = envs.SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS.get()
    if iters not in (4, 8, 12, 20):
        raise ValueError(
            f"unsupported gfx90a MHC Sinkhorn iterations: {iters}"
        )
    _jit_gfx90a_mhc_sinkhorn_module(iters).run(
        mixes, hc_scale, hc_base, pre, post, comb, eps
    )
    return pre, post, comb
