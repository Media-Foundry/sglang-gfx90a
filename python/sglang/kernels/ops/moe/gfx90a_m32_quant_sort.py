from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "gfx90a_m32_quant_sort_v2",
        cuda_files=["deepseek_v4/gfx90a_m32_quant_sort.cuh"],
        cuda_wrappers=[
            ("run_m32", "sglang::Gfx90aM32QuantSort::run"),
            ("run_m64", "sglang::Gfx90aM64QuantSort::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_m32_quant_sort(
    x: torch.Tensor, topk_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = torch.empty_like(x, dtype=torch.int8)
    m = x.shape[0]
    if m not in (32, 64) or topk_ids.shape != (m, 6):
        raise ValueError(f"expected M32/M64 top6, got {x.shape=} {topk_ids.shape=}")
    scales = torch.empty((m, 128), dtype=torch.float32, device=x.device)
    sorted_ids = torch.empty(m * 6 * 4, dtype=torch.int32, device=x.device)
    sorted_experts = torch.empty(m * 6, dtype=torch.int32, device=x.device)
    num_valid = torch.empty(2, dtype=torch.int32, device=x.device)
    run = _module().run_m32 if m == 32 else _module().run_m64
    run(
        x, topk_ids, q, scales, sorted_ids, sorted_experts, num_valid
    )
    return q, scales, sorted_ids, sorted_experts, num_valid
