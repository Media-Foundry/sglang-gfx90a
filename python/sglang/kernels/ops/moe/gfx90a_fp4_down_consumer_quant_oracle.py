from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module(m: int, intermediate_size: int, ctas_per_expert: int) -> Module:
    args = make_cpp_args(
        256, m, 6, 4096, intermediate_size, 4, 2, 8, ctas_per_expert
    )
    return load_jit(
        "gfx90a_fp4_down_consumer_quant_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_fp4_down_consumer_quant_oracle.cuh",
        ],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4DownConsumerQuantOracleKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_down_consumer_quant_oracle(
    intermediate: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    partial: torch.Tensor,
    *,
    ctas_per_expert: int,
) -> None:
    if not 1 <= ctas_per_expert <= 16:
        raise ValueError(f"unsupported ctas_per_expert={ctas_per_expert}")
    intermediate_size = intermediate.shape[-1]
    m = intermediate.shape[0]
    if m not in (32, 64):
        raise ValueError(f"unsupported token count={m}")
    if intermediate_size not in (256, 512):
        raise ValueError(f"unsupported intermediate_size={intermediate_size}")
    _jit_module(m, intermediate_size, ctas_per_expert).run(
        intermediate,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(
            256, 4096, intermediate_size // 32
        ),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk_weights,
        partial,
    )
