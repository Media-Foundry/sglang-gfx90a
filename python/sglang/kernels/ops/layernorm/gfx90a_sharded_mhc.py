from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_HC = 4
_HIDDEN_SHARD = 512
_MIX = 24


@cache_once
def _jit_gfx90a_sharded_mhc_module(iters: int) -> Module:
    fast_math = envs.SGLANG_DSV4_GFX90A_MHC_FAST_MATH.get()
    return load_jit(
        f"gfx90a_sharded_mhc_tp8_v1_i{iters}_{'fast' if fast_math else 'precise'}",
        cuda_files=["deepseek_v4/gfx90a_sharded_mhc.cuh"],
        cuda_wrappers=[
            ("stage1", "sglang::Gfx90aShardedMhcStage1Kernel::run"),
            ("stage2", "sglang::Gfx90aShardedMhcStage2Kernel::run"),
            ("stage3", "sglang::Gfx90aShardedMhcStage3Kernel::run"),
        ],
        extra_cuda_cflags=[
            "-O3",
            f"-DSGLANG_SHARDED_MHC_SINKHORN_ITERS={iters}",
        ]
        + (["-ffast-math"] if fast_math else []),
    )


def _get_sinkhorn_iters() -> int:
    iters = envs.SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS.get()
    if iters not in (4, 8, 12, 20):
        raise ValueError(f"unsupported gfx90a sharded-MHC Sinkhorn iterations: {iters}")
    return iters


def _check_gfx90a(tensors: tuple[torch.Tensor, ...]) -> None:
    if not torch.version.hip:
        raise RuntimeError("gfx90a sharded MHC requires a ROCm build of PyTorch")
    if not tensors:
        raise ValueError("at least one tensor is required")
    device = tensors[0].device
    if device.type != "cuda" or any(t.device != device for t in tensors):
        raise ValueError("all gfx90a sharded-MHC tensors must be on one CUDA/HIP device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all gfx90a sharded-MHC tensors must be contiguous")
    arch = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "").split(
        ":", 1
    )[0]
    if arch != "gfx90a":
        raise RuntimeError(f"gfx90a sharded MHC does not support device architecture {arch!r}")


def _expect(
    tensor: torch.Tensor,
    name: str,
    shape: tuple[int, ...],
    dtype: torch.dtype,
) -> None:
    if tensor.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def preload_gfx90a_sharded_mhc() -> None:
    _jit_gfx90a_sharded_mhc_module(_get_sinkhorn_iters())


def gfx90a_sharded_mhc_stage1(
    x_shard: torch.Tensor,
    residual_shard: torch.Tensor,
    previous_post: torch.Tensor,
    previous_comb: torch.Tensor,
    fn_shard: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(hc_post_shard, local_stats)`` before the first all-reduce.

    ``local_stats[:, :24]`` are rank-local pre-mix dot products and column 24
    is the local residual sum of squares.  The caller must sum-reduce all 25
    FP32 values over the TP8 group before calling :func:`stage2`.
    """

    tensors = (x_shard, residual_shard, previous_post, previous_comb, fn_shard)
    _check_gfx90a(tensors)
    m = x_shard.shape[0] if x_shard.ndim == 2 else -1
    if m <= 0:
        raise ValueError("x_shard must contain at least one token")
    _expect(x_shard, "x_shard", (m, _HIDDEN_SHARD), torch.bfloat16)
    _expect(
        residual_shard,
        "residual_shard",
        (m, _HC, _HIDDEN_SHARD),
        torch.bfloat16,
    )
    _expect(previous_post, "previous_post", (m, _HC), torch.float32)
    _expect(previous_comb, "previous_comb", (m, _HC, _HC), torch.float32)
    _expect(fn_shard, "fn_shard", (_MIX, _HC * _HIDDEN_SHARD), torch.float16)

    hc_post_shard = torch.empty_like(residual_shard)
    local_stats = torch.empty((m, _MIX + 1), dtype=torch.float32, device=x_shard.device)
    _jit_gfx90a_sharded_mhc_module(_get_sinkhorn_iters()).stage1(
        x_shard,
        residual_shard,
        previous_post,
        previous_comb,
        fn_shard,
        hc_post_shard,
        local_stats,
    )
    return hc_post_shard, local_stats


def gfx90a_sharded_mhc_stage2(
    hc_post_shard: torch.Tensor,
    global_stats: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Consume sum-reduced stage-1 stats.

    Returns ``(post, comb, y_rounded_shard, local_y_sumsq)``.  The caller must
    sum-reduce ``local_y_sumsq`` over TP8 before invoking :func:`stage3`.
    """

    tensors = (hc_post_shard, global_stats, hc_scale, hc_base)
    _check_gfx90a(tensors)
    m = hc_post_shard.shape[0] if hc_post_shard.ndim == 3 else -1
    if m <= 0:
        raise ValueError("hc_post_shard must contain at least one token")
    _expect(
        hc_post_shard,
        "hc_post_shard",
        (m, _HC, _HIDDEN_SHARD),
        torch.bfloat16,
    )
    _expect(global_stats, "global_stats", (m, _MIX + 1), torch.float32)
    _expect(hc_scale, "hc_scale", (3,), torch.float32)
    _expect(hc_base, "hc_base", (_MIX,), torch.float32)

    post = torch.empty((m, _HC), dtype=torch.float32, device=hc_post_shard.device)
    comb = torch.empty(
        (m, _HC, _HC), dtype=torch.float32, device=hc_post_shard.device
    )
    y_rounded_shard = torch.empty(
        (m, _HIDDEN_SHARD), dtype=torch.bfloat16, device=hc_post_shard.device
    )
    local_y_sumsq = torch.empty((m,), dtype=torch.float32, device=hc_post_shard.device)
    _jit_gfx90a_sharded_mhc_module(_get_sinkhorn_iters()).stage2(
        hc_post_shard,
        global_stats,
        hc_scale,
        hc_base,
        y_rounded_shard,
        post,
        comb,
        local_y_sumsq,
        float(rms_eps),
        float(sinkhorn_eps),
        float(post_multiplier),
    )
    return post, comb, y_rounded_shard, local_y_sumsq


def gfx90a_sharded_mhc_stage3(
    y_rounded_shard: torch.Tensor,
    global_y_sumsq: torch.Tensor,
    norm_weight_shard: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor:
    """RMS-normalize a hidden shard using sum-reduced full-hidden statistics."""

    tensors = (y_rounded_shard, global_y_sumsq, norm_weight_shard)
    _check_gfx90a(tensors)
    m = y_rounded_shard.shape[0] if y_rounded_shard.ndim == 2 else -1
    if m <= 0:
        raise ValueError("y_rounded_shard must contain at least one token")
    _expect(
        y_rounded_shard,
        "y_rounded_shard",
        (m, _HIDDEN_SHARD),
        torch.bfloat16,
    )
    _expect(global_y_sumsq, "global_y_sumsq", (m,), torch.float32)
    _expect(
        norm_weight_shard,
        "norm_weight_shard",
        (_HIDDEN_SHARD,),
        torch.bfloat16,
    )

    layer_input_shard = torch.empty_like(y_rounded_shard)
    _jit_gfx90a_sharded_mhc_module(_get_sinkhorn_iters()).stage3(
        y_rounded_shard,
        global_y_sumsq,
        norm_weight_shard,
        layer_input_shard,
        float(norm_eps),
    )
    return layer_input_shard
