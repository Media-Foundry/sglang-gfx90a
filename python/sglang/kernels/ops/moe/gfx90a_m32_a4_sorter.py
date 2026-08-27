from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "gfx90a_m32_a4_sorter_v2",
        cuda_files=["deepseek_v4/gfx90a_m32_a4_sorter.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aM32A4Sorter::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_m32_a4_sorter(
    topk_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sorted_ids = torch.empty(768, dtype=torch.int32, device=topk_ids.device)
    sorted_experts = torch.empty(192, dtype=torch.int32, device=topk_ids.device)
    num_valid = torch.empty(2, dtype=torch.int32, device=topk_ids.device)
    _module().run(topk_ids, sorted_ids, sorted_experts, num_valid)
    return sorted_ids, sorted_experts, num_valid
