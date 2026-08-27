from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_attn_q_dual_gemv",
        cuda_files=["deepseek_v4/gfx90a_attn_q_dual_gemv.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aAttnQDualGemvKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_attn_q_dual_gemv(
    x: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    out0 = torch.empty((1, 8192), dtype=torch.bfloat16, device=x.device)
    out1 = torch.empty_like(out0)
    _jit_module().run(x, w0, w1, out0, out1)
    return out0, out1
