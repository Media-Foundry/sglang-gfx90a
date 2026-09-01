"""Standalone wrapper for the gfx90a MFMA64 gate/quant I32-owner oracle.

This module is intentionally not imported by a model runner or production
selector.  The debug entry materializes the BF16 gate intermediate; the timing
entry writes only the group-32 INT8 values and FP32 scales consumed by stage 2.
"""

from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


@cache_once
def _jit_mfma64_gate_quant(
    e: int,
    m: int,
    topk: int,
    intermediate_size: int,
    hidden_size: int,
    blocks: int,
    split: int,
):
    args = make_cpp_args(
        e, m, topk, intermediate_size, hidden_size, blocks, split
    )
    return load_jit(
        "gfx90a_fp4_mfma64_gate_quant_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_mfma64_gate_quant_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_quant",
                f"sglang::Gfx90aFp4Mfma64GateQuantOracle<{args}>::run_quant",
            ),
            (
                "run_debug",
                f"sglang::Gfx90aFp4Mfma64GateQuantOracle<{args}>::run_debug",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_mfma64_gate_quant_oracle(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    topk: int = 6,
    limit: float = 10.0,
    blocks: int = 416,
    split: int = 4,
    debug_intermediate: bool = False,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor]:
    """Run the I32-owner gate epilogue.

    Weights must use the raw production layout ``[E, 2I, K/2]`` with the
    established shuffled E8M0 scale view ``[E, 2I, K/32]``.
    """

    m, hidden_size = xq.shape
    e, two_i, packed_k = weight.shape
    intermediate_size = two_i // 2
    assert packed_k * 2 == hidden_size
    assert intermediate_size % 32 == 0
    assert xq.dtype == torch.int8 and xq.is_contiguous()
    assert x_scale.shape == (m, hidden_size // 32)
    assert x_scale.dtype == torch.float32 and x_scale.is_contiguous()
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert weight_scale.numel() == e * two_i * (hidden_size // 32)
    assert sorted_ids.dtype == torch.int32 and sorted_ids.is_contiguous()
    assert sorted_expert_ids.dtype == torch.int32
    assert num_valid_ids.shape == (2,) and num_valid_ids.dtype == torch.int32
    assert topk == 6
    assert split == 4

    output_q = torch.empty(
        (m, topk, intermediate_size), dtype=torch.int8, device=xq.device
    )
    output_scale = torch.empty(
        (m, topk, intermediate_size // 32),
        dtype=torch.float32,
        device=xq.device,
    )
    intermediate = (
        torch.empty(
            (m, topk, intermediate_size),
            dtype=torch.bfloat16,
            device=xq.device,
        )
        if debug_intermediate
        else None
    )
    module = _jit_mfma64_gate_quant(
        e, m, topk, intermediate_size, hidden_size, blocks, split
    )
    scale_view = weight_scale.view(torch.uint8).reshape(
        e, two_i, hidden_size // 32
    )
    if intermediate is None:
        module.run_quant(
            xq,
            x_scale,
            weight,
            scale_view,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            output_q,
            output_scale,
            float(limit),
        )
    else:
        module.run_debug(
            xq,
            x_scale,
            weight,
            scale_view,
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            intermediate,
            output_q,
            output_scale,
            float(limit),
        )
    return intermediate, output_q, output_scale


__all__ = ["gfx90a_fp4_mfma64_gate_quant_oracle"]
