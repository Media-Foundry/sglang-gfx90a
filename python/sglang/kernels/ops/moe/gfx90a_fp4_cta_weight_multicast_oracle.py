from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


E, M, T, H, I, N = 256, 64, 6, 4096, 512, 4096
GATE_BLOCKS, DOWN_BLOCKS = 2080, 832


@cache_once
def _jit_module() -> Module:
    if not torch.version.hip:
        raise RuntimeError("M64 CTA multicast oracle requires ROCm")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
        ":", 1
    )[0]
    if arch != "gfx90a":
        raise RuntimeError(f"M64 CTA multicast oracle requires gfx90a, got {arch!r}")
    args = make_cpp_args(E, M, T, I, H, GATE_BLOCKS, DOWN_BLOCKS)
    return load_jit(
        "gfx90a_fp4_cta_weight_multicast_oracle_v1",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_cta_weight_multicast_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_gate",
                f"sglang::Gfx90aFp4CtaWeightMulticastOracle<{args}>::run_gate",
            ),
            (
                "run_down",
                f"sglang::Gfx90aFp4CtaWeightMulticastOracle<{args}>::run_down",
            ),
        ],
        extra_cuda_cflags=["-O3", "-std=c++20"],
    )


def preload() -> None:
    """Compile the oracle before graph capture; never called by production."""
    _jit_module()


def run_gate(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    descriptor_experts: torch.Tensor,
    descriptor_starts: torch.Tensor,
    descriptor_counts: torch.Tensor,
    num_descriptors: torch.Tensor,
    out: torch.Tensor,
    limit: float = 10.0,
) -> None:
    _jit_module().run_gate(
        xq,
        x_scale,
        weight,
        weight_scale,
        sorted_ids,
        descriptor_experts,
        descriptor_starts,
        descriptor_counts,
        num_descriptors,
        out,
        limit,
    )


def run_down(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    descriptor_experts: torch.Tensor,
    descriptor_starts: torch.Tensor,
    descriptor_counts: torch.Tensor,
    num_descriptors: torch.Tensor,
    topk_weights: torch.Tensor,
    partial: torch.Tensor,
) -> None:
    _jit_module().run_down(
        xq,
        x_scale,
        weight,
        weight_scale,
        sorted_ids,
        descriptor_experts,
        descriptor_starts,
        descriptor_counts,
        num_descriptors,
        topk_weights,
        partial,
    )


__all__ = ["preload", "run_gate", "run_down"]
