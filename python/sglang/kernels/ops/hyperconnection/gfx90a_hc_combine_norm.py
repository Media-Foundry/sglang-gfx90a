from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "gfx90a_qwen_hc_combine_norm",
        cuda_files=["hyperconnection/gfx90a_hc_combine_norm.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aQwenHcCombineNorm::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qwen_hc_combine_norm(
    block_output: torch.Tensor,
    residual: torch.Tensor,
    gate_partials: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined = torch.empty_like(residual)
    normed = torch.empty_like(residual)
    _module().run(
        block_output,
        residual,
        gate_partials,
        norm_weight,
        combined,
        normed,
        eps,
    )
    return combined, normed
