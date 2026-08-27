from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    args = make_cpp_args(256, 32, 6, 256, 4096, 4, 4, 2)
    return load_jit(
        "gfx90a_fp4_gate_wave_quant_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_fp4_gate_wave_quant_oracle.cuh",
        ],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4GateWaveQuantOracleKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_gate_wave_quant_oracle(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    intermediate: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    limit: float,
) -> None:
    _jit_module().run(
        xq,
        x_scale,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(256, 512, 128),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        intermediate,
        output_q,
        output_scale,
        limit,
    )
