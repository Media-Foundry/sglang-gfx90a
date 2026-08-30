from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_packed_weight_roofline",
        cuda_files=["deepseek_v4/gfx90a_packed_weight_roofline.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aPackedWeightRoofline::run"),
        ],
        extra_cuda_cflags=["-O3"],
    )


def packed_weight_roofline(
    weights: torch.Tensor,
    order: torch.Tensor,
    order_len: torch.Tensor,
    checksum: torch.Tensor,
) -> None:
    _jit_module().run(weights, order, order_len, checksum)
