from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "gfx90a_grouped_router_wave64_v4",
        cuda_files=["moe/gfx90a_grouped_router.cuh"],
        cuda_wrappers=[
            ("run_bf16", "sglang::Gfx90aGroupedRouterKernel::run"),
            ("run_fp32", "sglang::Gfx90aGroupedRouterFp32BiasKernel::run"),
            ("sqrt_bf16", "sglang::Gfx90aSqrtSoftplusRouterKernel::run"),
            (
                "sqrt_fp32",
                "sglang::Gfx90aSqrtSoftplusRouterFp32BiasKernel::run",
            ),
            (
                "sqrt_bf16_bf16",
                "sglang::Gfx90aSqrtSoftplusRouterBf16Kernel::run",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def preload_gfx90a_router() -> None:
    _module()


def gfx90a_grouped_router(
    scores: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not torch.version.hip
        or scores.shape != (1, 256)
        or scores.dtype != torch.float32
        or bias.shape != (256,)
        or bias.dtype not in (torch.bfloat16, torch.float32)
        or not scores.is_contiguous()
        or not bias.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(scores.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None
    weights = torch.empty((1, 6), dtype=torch.float32, device=scores.device)
    indices = torch.empty((1, 6), dtype=torch.int32, device=scores.device)
    run = _module().run_bf16 if bias.dtype == torch.bfloat16 else _module().run_fp32
    run(
        scores,
        bias,
        weights,
        indices,
        float(routed_scaling_factor),
        bool(apply_scale),
    )
    return weights, indices


def gfx90a_sqrtsoftplus_router(
    scores: torch.Tensor,
    bias: torch.Tensor,
    routed_scaling_factor: float,
    apply_scale: bool,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        scores.shape != (1, 256)
        or scores.dtype not in (torch.bfloat16, torch.float32)
        or bias.shape != (256,)
        or bias.dtype not in (torch.bfloat16, torch.float32)
        or not scores.is_contiguous()
        or not bias.is_contiguous()
    ):
        return None
    weights = torch.empty((1, 6), dtype=torch.float32, device=scores.device)
    indices = torch.empty((1, 6), dtype=torch.int32, device=scores.device)
    if scores.dtype == torch.bfloat16 and bias.dtype == torch.bfloat16:
        run = _module().sqrt_bf16_bf16
    elif scores.dtype == torch.float32 and bias.dtype == torch.bfloat16:
        run = _module().sqrt_bf16
    elif scores.dtype == torch.float32 and bias.dtype == torch.float32:
        run = _module().sqrt_fp32
    else:
        return None
    run(
        scores,
        bias,
        weights,
        indices,
        float(routed_scaling_factor),
        bool(apply_scale),
    )
    return weights, indices
