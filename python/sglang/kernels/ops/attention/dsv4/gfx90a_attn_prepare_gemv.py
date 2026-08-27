from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_attn_prepare_gemv_n4160",
        cuda_files=["deepseek_v4/gfx90a_attn_prepare_gemv.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aAttnPrepareGemvKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_attn_prepare_gemv(
    x: torch.Tensor,
    wqkv: torch.Tensor,
    wcore: torch.Tensor,
    windex: torch.Tensor,
    wweights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    out_qkv = torch.empty((1, 1536), dtype=torch.bfloat16, device=x.device)
    out_core = torch.empty((1, 2048), dtype=torch.float32, device=x.device)
    out_index = torch.empty((1, 512), dtype=torch.float32, device=x.device)
    out_weights = torch.empty((1, 64), dtype=torch.bfloat16, device=x.device)
    _jit_module().run(
        x,
        wqkv,
        wcore,
        windex,
        wweights,
        out_qkv,
        out_core,
        out_index,
        out_weights,
    )
    return out_qkv, out_core, out_index, out_weights
