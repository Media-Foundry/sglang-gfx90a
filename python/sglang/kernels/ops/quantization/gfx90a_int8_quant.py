from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_int8_group32_quant() -> Module:
    return load_jit(
        "gfx90a_int8_group32_quant",
        cuda_files=["deepseek_v4/gfx90a_int8_quant.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aInt8Group32QuantKernel::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_int8_group32_quant(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        x.device.type != "cuda"
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.numel() % 32 != 0
    ):
        raise ValueError(
            "gfx90a group32 quant expects contiguous CUDA BF16 with numel % 32 == 0"
        )
    groups = x.numel() // 32
    q = torch.empty_like(x, dtype=torch.int8)
    scales = torch.empty(groups, dtype=torch.float32, device=x.device)
    _jit_gfx90a_int8_group32_quant().run(
        x.view(groups, 32), q.view(groups, 32), scales
    )
    return q, scales.view(*x.shape[:-1], x.shape[-1] // 32)
