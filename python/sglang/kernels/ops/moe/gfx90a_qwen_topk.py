from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "gfx90a_qwen_topk",
        cuda_files=["moe/gfx90a_qwen_topk.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aQwenTopk::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qwen_topk(
    logits: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor
) -> None:
    _module().run(logits, weights, ids)
