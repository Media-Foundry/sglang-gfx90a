from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_tp8_shared_gate_round_module() -> Module:
    import logging

    args = make_cpp_args(512, 4096, 1, 2, 4, True)
    module = load_jit(
        "dsv4_tp8_shared_gate_torch_round_oracle", *args,
        cuda_files=["gemm/gfx90a_bf16_gated_gemv.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aBf16GatedGemvKernel<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )
    logging.getLogger(__name__).info(
        "DSV4 TP8 C1 shared gate round kernel selected: N512/K4096/R1/U2/W4"
    )
    return module


@cache_once
def _jit_gfx90a_bf16_gated_gemv_module(n: int, k: int) -> Module:
    rows, unroll, waves = (2, 1, 8)
    args = make_cpp_args(n, k, rows, unroll, waves)
    return load_jit(
        "gfx90a_bf16_gated_gemv",
        *args,
        cuda_files=["gemm/gfx90a_bf16_gated_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aBf16GatedGemvKernel<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_wave64_bf16_gated_gemv(
    x: torch.Tensor, weight: torch.Tensor, swiglu_limit: float
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or x.shape[0] != 1
        or weight.ndim != 2
        or x.shape[1] != weight.shape[1]
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or tuple(weight.shape) != (1024, 4096)
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None

    out = torch.empty((1, weight.shape[0] // 2), dtype=x.dtype, device=x.device)
    _jit_gfx90a_bf16_gated_gemv_module(weight.shape[0], weight.shape[1]).run(
        x, weight, out, float(swiglu_limit)
    )
    return out
