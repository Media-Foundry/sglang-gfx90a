from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(256, 32, 6, 4096, 256, 4, 2, 8, 832, 64)
    return load_jit(
        "gfx90a_fp4_hot_cache_down_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_fp4_hot_cache_oracle.cuh",
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4HotCacheDownOracleKernel<{args}>::run_partial",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_hot_cache_down_partial_oracle(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    hot_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_to_cache: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    """Run the standalone TP8 N64 mixed packed/prepacked w2 oracle."""
    _jit_module().run_partial(
        xq,
        x_scale,
        packed_weight.view(torch.uint8),
        hot_weight,
        weight_scale.view(torch.uint8).reshape(256, 4096, 8),
        expert_to_cache,
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk_weights,
        partial,
    )
