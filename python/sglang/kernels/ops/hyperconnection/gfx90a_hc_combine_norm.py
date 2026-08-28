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
    batch = block_output.shape[0]
    if (
        batch not in (1, 16, 32)
        or block_output.shape != (batch, 2560)
        or residual.shape != (batch, 10240)
        or gate_partials.shape != (batch, 8, 4)
        or norm_weight.shape != (10240,)
    ):
        raise ValueError(
            "gfx90a Qwen HC combine+norm requires B=1/16/32, "
            "block=[B,2560], residual=[B,10240], gate=[B,8,4], weight=[10240]"
        )
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
