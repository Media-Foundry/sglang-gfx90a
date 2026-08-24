import functools
import importlib
import logging
import math
import threading
from typing import Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import is_arch_support_pdl
from sglang.srt.distributed.device_communicators.pynccl_allocator import (
    use_symmetric_memory,
)
from sglang.srt.distributed.parallel_state import get_tp_group
from sglang.srt.environ import envs
from sglang.srt.layers.attention.dsa.utils import is_dsa_prefill_cp_round_robin_split
from sglang.srt.layers.dp_attention import is_allocation_symmetric
from sglang.srt.layers.utils.common import strict_contiguous

logger = logging.getLogger(__name__)

# This module is imported during model-registry discovery. Do not import the real
# TileLang package here: it loads native CUDA stubs. The proxy below lets
# module-level @tilelang.jit declarations parse, then imports and applies real
# TileLang only when a TileLang MHC kernel is actually called.
_real_tilelang = None
_real_T = None
_tilelang_load_lock = threading.Lock()


class _LazyTilelangAttr:
    def __init__(self, path: Tuple[str, ...] = ()):
        self.path = path

    def __getattr__(self, name):
        return _LazyTilelangAttr((*self.path, name))

    def __call__(self, *_args, **_kwargs):
        return _LazyTilelangAttr(self.path)


def _resolve_lazy_tilelang_value(value):
    if isinstance(value, _LazyTilelangAttr):
        obj = _load_tilelang()
        for name in value.path:
            obj = getattr(obj, name)
        return obj
    if isinstance(value, dict):
        return {
            _resolve_lazy_tilelang_value(k): _resolve_lazy_tilelang_value(v)
            for k, v in value.items()
        }
    # Keep list/tuple support so future TileLang jit kwargs such as out_idx=[...]
    # can use lazy TileLang enum values without changing the proxy.
    if isinstance(value, list):
        return [_resolve_lazy_tilelang_value(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_resolve_lazy_tilelang_value(v) for v in value)
    return value


def _load_tilelang():
    global _real_tilelang, _real_T, tilelang, T
    if _real_tilelang is None:
        with _tilelang_load_lock:
            if _real_tilelang is None:
                try:
                    new_tilelang = importlib.import_module("tilelang")
                    new_T = importlib.import_module("tilelang.language")
                except ImportError as exc:
                    raise RuntimeError(
                        "tilelang is not installed; this kernel cannot run on the current platform"
                    ) from exc
                new_tilelang.set_log_level("WARNING")
                tilelang = new_tilelang
                T = new_T
                _real_T = new_T
                _real_tilelang = new_tilelang
    return _real_tilelang


class _LazyTilelang:
    PassConfigKey = _LazyTilelangAttr(("PassConfigKey",))
    layout = _LazyTilelangAttr(("layout",))

    def jit(self, func=None, **jit_kwargs):
        def decorate(fn):
            compiled = None
            compile_lock = threading.Lock()

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                nonlocal compiled
                if compiled is None:
                    with compile_lock:
                        if compiled is None:
                            real_tilelang = _load_tilelang()
                            real_kwargs = _resolve_lazy_tilelang_value(jit_kwargs)
                            compiled = real_tilelang.jit(**real_kwargs)(fn)
                return compiled(*args, **kwargs)

            return wrapper

        if callable(func):
            return decorate(func)
        return decorate

    def __getattr__(self, name):
        return _LazyTilelangAttr((name,))


tilelang = _LazyTilelang()
T = _LazyTilelangAttr()
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
}

FP8 = "float8_e4m3"
BF16 = "bfloat16"
FP32 = "float32"
INT32 = "int32"


@triton.jit
def _gfx90a_mhc_pre_mix_kernel(
    residual,
    fn,
    mixes,
    k: tl.constexpr,
    n: tl.constexpr,
    rms_eps: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    sq_sum = tl.zeros((), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(residual + token_id * k + offs_k).to(tl.float32)
        w = tl.load(
            fn + offs_n[:, None] * k + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        )
        acc += tl.sum(w * x[None, :], axis=1)
        sq_sum += tl.sum(x * x, axis=0)
    scale = tl.rsqrt(sq_sum / k + rms_eps)
    tl.store(mixes + token_id * n + offs_n, acc * scale, mask=offs_n < n)


@triton.jit
def _gfx90a_mhc_mix_kernel(
    residual,
    fn,
    rsqrt,
    mixes,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(residual + token_id * k + offs_k).to(tl.float32)
        w = tl.load(
            fn + offs_n[:, None] * k + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        )
        acc += tl.sum(w * x[None, :], axis=1)
    scale = tl.load(rsqrt + token_id)
    tl.store(mixes + token_id * n + offs_n, acc * scale, mask=offs_n < n)


@triton.jit
def _gfx90a_mhc_mix_partials_kernel(
    residual,
    fn,
    rms_partials,
    mixes,
    k: tl.constexpr,
    n: tl.constexpr,
    rms_eps: tl.constexpr,
    NUM_PARTIALS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(residual + token_id * k + offs_k).to(tl.float32)
        w = tl.load(
            fn + offs_n[:, None] * k + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        )
        acc += tl.sum(w * x[None, :], axis=1)
    offs_p = tl.arange(0, NUM_PARTIALS)
    sq_sum = tl.sum(
        tl.load(rms_partials + token_id * NUM_PARTIALS + offs_p), axis=0
    )
    scale = tl.rsqrt(sq_sum / k + rms_eps)
    tl.store(mixes + token_id * n + offs_n, acc * scale, mask=offs_n < n)


@triton.jit
def _gfx90a_mhc_mix_splitk_stage0_kernel(
    residual,
    fn,
    dot_partials,
    k: tl.constexpr,
    SPLITS: tl.constexpr,
    CHUNK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    rows = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    split = tl.program_id(1)
    token_id = tl.program_id(2)
    acc = tl.zeros((BLOCK_N,), tl.float32)
    for local_k in tl.static_range(0, CHUNK_K, BLOCK_K):
        offs_k = split * CHUNK_K + local_k + tl.arange(0, BLOCK_K)
        x = tl.load(residual + token_id * k + offs_k).to(tl.float32)
        w = tl.load(
            fn + rows[:, None] * k + offs_k[None, :],
            mask=rows[:, None] < 24,
            other=0.0,
        )
        acc += tl.sum(w * x[None, :], axis=1)
    tl.store(
        dot_partials + (token_id * 24 + rows) * SPLITS + split,
        acc,
        mask=rows < 24,
    )


@triton.jit
def _gfx90a_mhc_mix_splitk_stage1_kernel(
    dot_partials,
    rms_partials,
    mixes,
    k: tl.constexpr,
    SPLITS: tl.constexpr,
    NUM_RMS_PARTIALS: tl.constexpr,
    rms_eps: tl.constexpr,
):
    token_id = tl.program_id(0)
    rows = tl.arange(0, 32)
    splits = tl.arange(0, SPLITS)
    values = tl.load(
        dot_partials
        + (token_id * 24 + rows[:, None]) * SPLITS
        + splits[None, :],
        mask=rows[:, None] < 24,
        other=0.0,
    )
    dot = tl.sum(values, axis=1)
    rms_offsets = tl.arange(0, NUM_RMS_PARTIALS)
    sq_sum = tl.sum(
        tl.load(rms_partials + token_id * NUM_RMS_PARTIALS + rms_offsets), axis=0
    )
    scale = tl.rsqrt(sq_sum / k + rms_eps)
    tl.store(mixes + token_id * 24 + rows, dot * scale, mask=rows < 24)


@triton.jit
def _gfx90a_mhc_splitk_fused_tail_kernel(
    dot_partials,
    rms_partials,
    residual,
    hc_scale,
    hc_base,
    norm_weight,
    post,
    comb,
    out,
    eps: tl.constexpr,
    norm_eps: tl.constexpr,
    SINKHORN_ITERS: tl.constexpr,
):
    token_id = tl.program_id(0)
    splits = tl.arange(0, 8)
    rms_offsets = tl.arange(0, 64)
    sq_sum = tl.sum(tl.load(rms_partials + token_id * 64 + rms_offsets))
    pre_scale = tl.rsqrt(sq_sum / 16384.0 + eps)

    offs4 = tl.arange(0, 4)
    pre_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + offs4[:, None]) * 8
            + splits[None, :]
        ),
        axis=1,
    )
    post_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + 4 + offs4[:, None]) * 8
            + splits[None, :]
        ),
        axis=1,
    )
    pre_v = tl.sigmoid(
        pre_dot * pre_scale * tl.load(hc_scale) + tl.load(hc_base + offs4)
    ) + eps
    post_v = 2.0 * tl.sigmoid(
        post_dot * pre_scale * tl.load(hc_scale + 1)
        + tl.load(hc_base + 4 + offs4)
    )
    tl.store(post + token_id * 4 + offs4, post_v)

    offs16 = tl.arange(0, 16)
    comb_dot = tl.sum(
        tl.load(
            dot_partials
            + (token_id * 24 + 8 + offs16[:, None]) * 8
            + splits[None, :]
        ),
        axis=1,
    )
    matrix = tl.reshape(
        comb_dot * pre_scale * tl.load(hc_scale + 2)
        + tl.load(hc_base + 8 + offs16),
        (4, 4),
    )
    matrix = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = matrix / tl.sum(matrix, axis=1)[:, None] + eps
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    for _ in tl.static_range(0, SINKHORN_ITERS - 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + eps)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    tl.store(comb + token_id * 16 + offs16, tl.reshape(matrix, (16,)))

    hidden = tl.arange(0, 4096)
    channels = tl.arange(0, 4)
    residual_v = tl.load(
        residual
        + token_id * 16384
        + channels[:, None] * 4096
        + hidden[None, :]
    ).to(tl.float32)
    weighted = tl.sum(pre_v[:, None] * residual_v, axis=0)
    rounded = weighted.to(tl.bfloat16).to(tl.float32)
    inv_rms = tl.rsqrt(tl.sum(rounded * rounded) / 4096.0 + norm_eps)
    weight = tl.load(norm_weight + hidden).to(tl.float32)
    tl.store(out + token_id * 4096 + hidden, rounded * inv_rms * weight)


@triton.jit
def _gfx90a_mhc_mix_bf16_dot_kernel(
    residual,
    fn_bf16,
    rsqrt,
    mixes,
    k: tl.constexpr,
    n: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """MFMA-backed MHC GEMV; only column zero of the 16-wide dot is used."""
    token_id = tl.program_id(1)
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    acc = tl.zeros((BLOCK_N, BLOCK_N), tl.float32)
    for k_start in tl.static_range(0, k, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)
        x = tl.load(residual + token_id * k + offs_k).to(tl.bfloat16)
        w = tl.load(
            fn_bf16 + offs_n[:, None] * k + offs_k[None, :],
            mask=offs_n[:, None] < n,
            other=0.0,
        ).to(tl.bfloat16)
        # gfx90a MFMA requires N >= 16. Replicating the vector across a
        # 16-column tile trades redundant arithmetic for tensor-core execution;
        # weights are still read exactly once and FP32 accumulation is retained.
        x_tile = tl.broadcast_to(x[:, None], (BLOCK_K, BLOCK_N))
        acc += tl.dot(w, x_tile)
    scale = tl.load(rsqrt + token_id)
    # Every MFMA output column is identical; Triton does not support scalar
    # column indexing on this 2-D block, so reduce and divide by the tile width.
    result = tl.sum(acc, axis=1) / BLOCK_N
    tl.store(
        mixes + token_id * n + offs_n,
        result * scale,
        mask=offs_n < n,
    )


@triton.jit
def _gfx90a_mhc_rsqrt_kernel(
    residual,
    rsqrt,
    k: tl.constexpr,
    rms_eps: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs_k = tl.arange(0, BLOCK_K)
    x = tl.load(
        residual + token_id * k + offs_k, mask=offs_k < k, other=0.0
    ).to(tl.float32)
    sq_sum = tl.sum(x * x, axis=0)
    tl.store(rsqrt + token_id, tl.rsqrt(sq_sum / k + rms_eps))


def gfx90a_mhc_pre_mix_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or residual.ndim != 3
        or residual.shape[0] < 1
        or residual.shape[1:] != (4, 4096)
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or fn.shape != (24, 16384)
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    num_tokens = residual.shape[0]
    residual_flat = residual.flatten(1)
    if envs.SGLANG_DSV4_GFX90A_FUSED_RMS_MHC_PRE_MIX.get():
        mixes = torch.empty(
            (num_tokens, 1, 24), dtype=torch.float32, device=residual.device
        )
        block_n = 2
        _gfx90a_mhc_pre_mix_kernel[(triton.cdiv(24, block_n), num_tokens)](
            residual_flat,
            fn,
            mixes,
            k=16384,
            n=24,
            rms_eps=rms_eps,
            BLOCK_N=block_n,
            BLOCK_K=512,
            num_warps=2,
        )
        return mixes

    rsqrt = torch.empty((num_tokens,), dtype=torch.float32, device=residual.device)
    _gfx90a_mhc_rsqrt_kernel[(num_tokens,)](
        residual_flat,
        rsqrt,
        k=16384,
        rms_eps=rms_eps,
        BLOCK_K=16384,
        num_warps=8,
    )
    mixes = torch.empty(
        (num_tokens, 1, 24), dtype=torch.float32, device=residual.device
    )
    block_n = 1
    # K=2048 is slightly faster standalone, but 1024 is the validated Mori
    # graph geometry. Keep the faster tile available for isolated A/B work.
    block_k = envs.SGLANG_DSV4_GFX90A_MHC_BLOCK_K.get()
    _gfx90a_mhc_mix_kernel[(triton.cdiv(24, block_n), num_tokens)](
        residual_flat,
        fn,
        rsqrt,
        mixes,
        k=16384,
        n=24,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=1,
    )
    return mixes


def gfx90a_mhc_pre_mix_from_partials_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    rms_partials: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or residual.ndim != 3
        or residual.shape[0] < 1
        or residual.shape[1:] != (4, 4096)
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or fn.shape != (24, 16384)
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or rms_partials.shape != (residual.shape[0], 64)
        or rms_partials.dtype != torch.float32
        or not rms_partials.is_contiguous()
    ):
        return None
    num_tokens = residual.shape[0]
    mixes = torch.empty(
        (num_tokens, 1, 24), dtype=torch.float32, device=residual.device
    )
    block_k = envs.SGLANG_DSV4_GFX90A_MHC_BLOCK_K.get()
    _gfx90a_mhc_mix_partials_kernel[(24, num_tokens)](
        residual.flatten(1),
        fn,
        rms_partials,
        mixes,
        k=16384,
        n=24,
        rms_eps=rms_eps,
        NUM_PARTIALS=64,
        BLOCK_N=1,
        BLOCK_K=block_k,
        num_warps=1,
    )
    return mixes


def gfx90a_mhc_pre_mix_splitk_from_partials_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    rms_partials: torch.Tensor,
    rms_eps: float,
    global_batch_size: int | None,
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or residual.ndim != 3
        or residual.shape[0] < 1
        or residual.shape[1:] != (4, 4096)
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or fn.shape != (24, 16384)
        or fn.dtype != torch.float32
        or not fn.is_contiguous()
        or rms_partials.shape != (residual.shape[0], 64)
        or rms_partials.dtype != torch.float32
        or not rms_partials.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    # TP-only decode has no Mori progress kernel to protect, so it may use the
    # 192-CTA scalar-row geometry that wins in isolation on a 104-CU GCD.
    if global_batch_size != 1:
        return None
    num_tokens = residual.shape[0]
    splits = 8
    block_n = (
        1
        if num_tokens == 1
        and envs.SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY.get()
        else 4
    )
    dot_partials = torch.empty(
        (num_tokens, 24, splits), dtype=torch.float32, device=residual.device
    )
    mixes = torch.empty(
        (num_tokens, 1, 24), dtype=torch.float32, device=residual.device
    )
    _gfx90a_mhc_mix_splitk_stage0_kernel[
        (triton.cdiv(24, block_n), splits, num_tokens)
    ](
        residual.flatten(1),
        fn,
        dot_partials,
        k=16384,
        SPLITS=splits,
        CHUNK_K=16384 // splits,
        BLOCK_N=block_n,
        BLOCK_K=1024,
        num_warps=1,
    )
    _gfx90a_mhc_mix_splitk_stage1_kernel[(num_tokens,)](
        dot_partials,
        rms_partials,
        mixes,
        k=16384,
        SPLITS=splits,
        NUM_RMS_PARTIALS=64,
        rms_eps=rms_eps,
        num_warps=1,
    )
    return mixes


def gfx90a_mhc_splitk_fused_tail_triton(
    residual: torch.Tensor,
    fn: torch.Tensor,
    fn_fp16: torch.Tensor | None,
    rms_partials: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    sinkhorn_eps: float,
    norm_eps: float,
    global_batch_size: int | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    num_tokens = residual.shape[0]
    if (
        global_batch_size != 1
        or residual.shape != (num_tokens, 4, 4096)
        or residual.dtype != torch.bfloat16
        or fn.shape != (24, 16384)
        or fn.dtype != torch.float32
        or rms_partials.shape != (num_tokens, 64)
        or rms_partials.dtype != torch.float32
        or hc_scale.shape != (3,)
        or hc_scale.dtype != torch.float32
        or hc_base.shape != (24,)
        or hc_base.dtype != torch.float32
        or norm_weight.shape != (4096,)
        or norm_weight.dtype != torch.bfloat16
        or not all(
            tensor.is_contiguous()
            for tensor in (
                residual,
                fn,
                rms_partials,
                hc_scale,
                hc_base,
                norm_weight,
            )
        )
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    mix_weight = fn
    if envs.SGLANG_DSV4_GFX90A_FP16_MHC_DOT.get():
        if (
            fn_fp16 is None
            or fn_fp16.shape != (24, 16384)
            or fn_fp16.dtype != torch.float16
            or not fn_fp16.is_contiguous()
        ):
            return None
        mix_weight = fn_fp16

    splits = 8
    block_n = (
        1
        if num_tokens == 1
        and envs.SGLANG_DSV4_GFX90A_MHC_TP_ONLY_GEOMETRY.get()
        else 4
    )
    dot_partials = torch.empty(
        (num_tokens, 24, splits), dtype=torch.float32, device=residual.device
    )
    post = torch.empty((num_tokens, 4), dtype=torch.float32, device=residual.device)
    comb = torch.empty(
        (num_tokens, 4, 4), dtype=torch.float32, device=residual.device
    )
    out = torch.empty(
        (num_tokens, 4096), dtype=torch.bfloat16, device=residual.device
    )
    _gfx90a_mhc_mix_splitk_stage0_kernel[
        (triton.cdiv(24, block_n), splits, num_tokens)
    ](
        residual.flatten(1),
        mix_weight,
        dot_partials,
        k=16384,
        SPLITS=splits,
        CHUNK_K=2048,
        BLOCK_N=block_n,
        BLOCK_K=1024,
        num_warps=1,
    )
    _gfx90a_mhc_splitk_fused_tail_kernel[(num_tokens,)](
        dot_partials,
        rms_partials,
        residual,
        hc_scale,
        hc_base,
        norm_weight,
        post,
        comb,
        out,
        eps=sinkhorn_eps,
        norm_eps=norm_eps,
        SINKHORN_ITERS=envs.SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS.get(),
        num_warps=8,
    )
    return post, comb, out


def gfx90a_mhc_pre_mix_bf16_dot_triton(
    residual: torch.Tensor,
    fn_bf16: torch.Tensor,
    rms_eps: float,
) -> torch.Tensor | None:
    """gfx90a MHC pre-mix using BF16 MFMA with FP32 accumulation."""
    if (
        not torch.version.hip
        or residual.ndim != 3
        or residual.shape[0] < 1
        or residual.shape[1:] != (4, 4096)
        or residual.dtype != torch.bfloat16
        or not residual.is_contiguous()
        or fn_bf16.shape != (24, 16384)
        or fn_bf16.dtype != torch.bfloat16
        or not fn_bf16.is_contiguous()
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None

    num_tokens = residual.shape[0]
    residual_flat = residual.flatten(1)
    rsqrt = torch.empty((num_tokens,), dtype=torch.float32, device=residual.device)
    _gfx90a_mhc_rsqrt_kernel[(num_tokens,)](
        residual_flat,
        rsqrt,
        k=16384,
        rms_eps=rms_eps,
        BLOCK_K=16384,
        num_warps=8,
    )
    mixes = torch.empty(
        (num_tokens, 1, 24), dtype=torch.float32, device=residual.device
    )
    _gfx90a_mhc_mix_bf16_dot_kernel[(2, num_tokens)](
        residual_flat,
        fn_bf16,
        rsqrt,
        mixes,
        k=16384,
        n=24,
        BLOCK_N=16,
        BLOCK_K=512,
        num_warps=4,
    )
    return mixes


@triton.jit
def _mhc_weighted_sum_kernel(
    x,
    pre,
    y,
    hidden_size: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_xc: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_pt: tl.constexpr,
    stride_pc: tl.constexpr,
    stride_yt: tl.constexpr,
    stride_yh: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    h_block = tl.program_id(1)
    offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    p0 = tl.load(pre + token_id * stride_pt + 0 * stride_pc)
    p1 = tl.load(pre + token_id * stride_pt + 1 * stride_pc)
    p2 = tl.load(pre + token_id * stride_pt + 2 * stride_pc)
    p3 = tl.load(pre + token_id * stride_pt + 3 * stride_pc)

    base = x + token_id * stride_xt + offs * stride_xh
    v0 = tl.load(base + 0 * stride_xc, mask=mask, other=0.0).to(tl.float32)
    v1 = tl.load(base + 1 * stride_xc, mask=mask, other=0.0).to(tl.float32)
    v2 = tl.load(base + 2 * stride_xc, mask=mask, other=0.0).to(tl.float32)
    v3 = tl.load(base + 3 * stride_xc, mask=mask, other=0.0).to(tl.float32)
    out = p0 * v0 + p1 * v1 + p2 * v2 + p3 * v3
    tl.store(y + token_id * stride_yt + offs * stride_yh, out, mask=mask)


@triton.jit
def _mhc_post_combine_kernel(
    x,
    residual,
    post,
    comb,
    out,
    rms_partials,
    hidden_size: tl.constexpr,
    stride_xt: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_rt: tl.constexpr,
    stride_ri: tl.constexpr,
    stride_rh: tl.constexpr,
    stride_pt: tl.constexpr,
    stride_po: tl.constexpr,
    stride_ct: tl.constexpr,
    stride_ci: tl.constexpr,
    stride_co: tl.constexpr,
    stride_ot: tl.constexpr,
    stride_oo: tl.constexpr,
    stride_oh: tl.constexpr,
    BLOCK_H: tl.constexpr,
    NUM_H_BLOCKS: tl.constexpr,
    WRITE_RMS_PARTIALS: tl.constexpr,
):
    token_id = tl.program_id(0)
    out_hc = tl.program_id(1)
    h_block = tl.program_id(2)
    offs = h_block * BLOCK_H + tl.arange(0, BLOCK_H)
    mask = offs < hidden_size

    post_v = tl.load(post + token_id * stride_pt + out_hc * stride_po)
    x_v = tl.load(x + token_id * stride_xt + offs * stride_xh, mask=mask, other=0.0).to(
        tl.float32
    )
    acc = post_v * x_v

    r_base = residual + token_id * stride_rt + offs * stride_rh
    c_base = comb + token_id * stride_ct + out_hc * stride_co
    c0 = tl.load(c_base + 0 * stride_ci)
    c1 = tl.load(c_base + 1 * stride_ci)
    c2 = tl.load(c_base + 2 * stride_ci)
    c3 = tl.load(c_base + 3 * stride_ci)
    r0 = tl.load(r_base + 0 * stride_ri, mask=mask, other=0.0).to(tl.float32)
    r1 = tl.load(r_base + 1 * stride_ri, mask=mask, other=0.0).to(tl.float32)
    r2 = tl.load(r_base + 2 * stride_ri, mask=mask, other=0.0).to(tl.float32)
    r3 = tl.load(r_base + 3 * stride_ri, mask=mask, other=0.0).to(tl.float32)
    acc += c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3

    tl.store(
        out + token_id * stride_ot + out_hc * stride_oo + offs * stride_oh,
        acc,
        mask=mask,
    )
    if WRITE_RMS_PARTIALS:
        rounded = acc.to(tl.bfloat16).to(tl.float32)
        partial = tl.sum(tl.where(mask, rounded * rounded, 0.0), axis=0)
        partial_offset = (
            (token_id * 4 + out_hc) * NUM_H_BLOCKS + h_block
        )
        tl.store(rms_partials + partial_offset, partial)


@triton.jit
def _gfx90a_mhc_rmsnorm_kernel(
    x,
    weight,
    out,
    hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    offs = tl.arange(0, BLOCK_H)
    mask = offs < hidden_size
    values = tl.load(
        x + token_id * hidden_size + offs, mask=mask, other=0.0
    ).to(tl.float32)
    rms = tl.rsqrt(tl.sum(values * values, axis=0) / hidden_size + eps)
    w = tl.load(weight + offs, mask=mask, other=0.0).to(tl.float32)
    tl.store(
        out + token_id * hidden_size + offs,
        values * rms * w,
        mask=mask,
    )


@triton.jit
def _gfx90a_mhc_weighted_rmsnorm_kernel(
    residual,
    pre,
    norm_weight,
    out,
    hidden_size: tl.constexpr,
    norm_eps: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    token_id = tl.program_id(0)
    h = tl.arange(0, BLOCK_H)
    mask = h < hidden_size
    base = residual + token_id * 4 * hidden_size + h
    acc = tl.zeros((BLOCK_H,), tl.float32)
    for channel in tl.static_range(0, 4):
        rv = tl.load(base + channel * hidden_size, mask=mask, other=0.0).to(
            tl.float32
        )
        pv = tl.load(pre + token_id * 4 + channel)
        acc += pv * rv
    y = acc.to(tl.bfloat16).to(tl.float32)
    sq = tl.sum(tl.where(mask, y * y, 0.0), axis=0)
    inv = tl.rsqrt(sq / hidden_size + norm_eps)
    w = tl.load(norm_weight + h, mask=mask, other=0.0).to(tl.float32)
    tl.store(out + token_id * hidden_size + h, y * inv * w, mask=mask)


def mhc_weighted_sum_triton(
    x: torch.Tensor,
    pre: torch.Tensor,
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 3
        or pre.ndim != 2
        or x.shape[1] != 4
        or pre.shape[1] != 4
        or x.shape[0] != pre.shape[0]
        or x.dtype != torch.bfloat16
    ):
        return None

    num_tokens, _, hidden_size = x.shape
    y = torch.empty((num_tokens, hidden_size), dtype=x.dtype, device=x.device)
    block_h = 256
    grid = (num_tokens, triton.cdiv(hidden_size, block_h))
    _mhc_weighted_sum_kernel[grid](
        x,
        pre,
        y,
        hidden_size,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        pre.stride(0),
        pre.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_H=block_h,
        num_warps=4,
    )
    return y


def gfx90a_mhc_weighted_rmsnorm_triton(
    residual: torch.Tensor,
    pre: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
) -> torch.Tensor | None:
    tokens = residual.shape[0]
    if (
        residual.shape != (tokens, 4, 4096)
        or pre.shape != (tokens, 4)
        or norm_weight.shape != (4096,)
        or residual.dtype != torch.bfloat16
        or pre.dtype != torch.float32
        or norm_weight.dtype != torch.bfloat16
        or not all(t.is_contiguous() for t in (residual, pre, norm_weight))
    ):
        return None
    out = torch.empty((tokens, 4096), dtype=torch.bfloat16, device=residual.device)
    _gfx90a_mhc_weighted_rmsnorm_kernel[(tokens,)](
        residual,
        pre,
        norm_weight,
        out,
        hidden_size=4096,
        norm_eps=norm_eps,
        BLOCK_H=4096,
        num_warps=8,
    )
    return out


def mhc_post_combine_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> torch.Tensor | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or residual.ndim != 3
        or post.ndim != 2
        or comb.ndim != 3
        or residual.shape[1] != 4
        or post.shape[1] != 4
        or comb.shape[1:] != (4, 4)
        or x.shape[0] != residual.shape[0]
        or x.shape[0] != post.shape[0]
        or x.shape[0] != comb.shape[0]
        or x.shape[1] != residual.shape[2]
        or x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
    ):
        return None

    num_tokens, hidden_size = x.shape
    out = torch.empty_like(residual)
    block_h = 256
    grid = (num_tokens, 4, triton.cdiv(hidden_size, block_h))
    _mhc_post_combine_kernel[grid](
        x,
        residual,
        post,
        comb,
        out,
        out,
        hidden_size,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post.stride(0),
        post.stride(1),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        NUM_H_BLOCKS=triton.cdiv(hidden_size, block_h),
        WRITE_RMS_PARTIALS=False,
        num_warps=4,
    )
    return out


def mhc_post_combine_rms_triton(
    x: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or residual.ndim != 3
        or post.ndim != 2
        or comb.ndim != 3
        or residual.shape[1:] != (4, 4096)
        or post.shape[1] != 4
        or comb.shape[1:] != (4, 4)
        or x.shape != (residual.shape[0], 4096)
        or x.shape[0] != post.shape[0]
        or x.shape[0] != comb.shape[0]
        or x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
    ):
        return None
    num_tokens, hidden_size = x.shape
    out = torch.empty_like(residual)
    block_h = 256
    num_h_blocks = triton.cdiv(hidden_size, block_h)
    rms_partials = torch.empty(
        (num_tokens, 4 * num_h_blocks), dtype=torch.float32, device=x.device
    )
    _mhc_post_combine_kernel[(num_tokens, 4, num_h_blocks)](
        x,
        residual,
        post,
        comb,
        out,
        rms_partials,
        hidden_size,
        x.stride(0),
        x.stride(1),
        residual.stride(0),
        residual.stride(1),
        residual.stride(2),
        post.stride(0),
        post.stride(1),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_H=block_h,
        NUM_H_BLOCKS=num_h_blocks,
        WRITE_RMS_PARTIALS=True,
        num_warps=4,
    )
    return out, rms_partials


@triton.jit
def _hc_split_sinkhorn4_kernel(
    mixes,
    hc_scale,
    hc_base,
    pre,
    post,
    comb,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_ps: tl.constexpr,
    stride_pv: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_ov: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cs: tl.constexpr,
    stride_ci: tl.constexpr,
    stride_cj: tl.constexpr,
    seq_len: tl.constexpr,
    eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid - b * seq_len
    mix = mixes + b * stride_mb + s * stride_ms

    scale0 = tl.load(hc_scale + 0)
    scale1 = tl.load(hc_scale + 1)
    scale2 = tl.load(hc_scale + 2)

    offs4 = tl.arange(0, 4)
    pre_v = (
        1.0
        / (
            1.0
            + tl.exp(
                -(
                    tl.load(mix + offs4) * scale0
                    + tl.load(hc_base + offs4)
                )
            )
        )
        + eps
    )
    post_v = (
        2.0
        / (
            1.0
            + tl.exp(
                -(
                    tl.load(mix + 4 + offs4) * scale1
                    + tl.load(hc_base + 4 + offs4)
                )
            )
        )
    )
    tl.store(pre + b * stride_pb + s * stride_ps + offs4 * stride_pv, pre_v)
    tl.store(post + b * stride_ob + s * stride_os + offs4 * stride_ov, post_v)

    c00 = tl.load(mix + 8) * scale2 + tl.load(hc_base + 8)
    c01 = tl.load(mix + 9) * scale2 + tl.load(hc_base + 9)
    c02 = tl.load(mix + 10) * scale2 + tl.load(hc_base + 10)
    c03 = tl.load(mix + 11) * scale2 + tl.load(hc_base + 11)
    c10 = tl.load(mix + 12) * scale2 + tl.load(hc_base + 12)
    c11 = tl.load(mix + 13) * scale2 + tl.load(hc_base + 13)
    c12 = tl.load(mix + 14) * scale2 + tl.load(hc_base + 14)
    c13 = tl.load(mix + 15) * scale2 + tl.load(hc_base + 15)
    c20 = tl.load(mix + 16) * scale2 + tl.load(hc_base + 16)
    c21 = tl.load(mix + 17) * scale2 + tl.load(hc_base + 17)
    c22 = tl.load(mix + 18) * scale2 + tl.load(hc_base + 18)
    c23 = tl.load(mix + 19) * scale2 + tl.load(hc_base + 19)
    c30 = tl.load(mix + 20) * scale2 + tl.load(hc_base + 20)
    c31 = tl.load(mix + 21) * scale2 + tl.load(hc_base + 21)
    c32 = tl.load(mix + 22) * scale2 + tl.load(hc_base + 22)
    c33 = tl.load(mix + 23) * scale2 + tl.load(hc_base + 23)

    m0 = tl.maximum(tl.maximum(c00, c01), tl.maximum(c02, c03))
    m1 = tl.maximum(tl.maximum(c10, c11), tl.maximum(c12, c13))
    m2 = tl.maximum(tl.maximum(c20, c21), tl.maximum(c22, c23))
    m3 = tl.maximum(tl.maximum(c30, c31), tl.maximum(c32, c33))
    c00 = tl.exp(c00 - m0)
    c01 = tl.exp(c01 - m0)
    c02 = tl.exp(c02 - m0)
    c03 = tl.exp(c03 - m0)
    c10 = tl.exp(c10 - m1)
    c11 = tl.exp(c11 - m1)
    c12 = tl.exp(c12 - m1)
    c13 = tl.exp(c13 - m1)
    c20 = tl.exp(c20 - m2)
    c21 = tl.exp(c21 - m2)
    c22 = tl.exp(c22 - m2)
    c23 = tl.exp(c23 - m2)
    c30 = tl.exp(c30 - m3)
    c31 = tl.exp(c31 - m3)
    c32 = tl.exp(c32 - m3)
    c33 = tl.exp(c33 - m3)

    r0 = c00 + c01 + c02 + c03
    r1 = c10 + c11 + c12 + c13
    r2 = c20 + c21 + c22 + c23
    r3 = c30 + c31 + c32 + c33
    c00 = c00 / r0 + eps
    c01 = c01 / r0 + eps
    c02 = c02 / r0 + eps
    c03 = c03 / r0 + eps
    c10 = c10 / r1 + eps
    c11 = c11 / r1 + eps
    c12 = c12 / r1 + eps
    c13 = c13 / r1 + eps
    c20 = c20 / r2 + eps
    c21 = c21 / r2 + eps
    c22 = c22 / r2 + eps
    c23 = c23 / r2 + eps
    c30 = c30 / r3 + eps
    c31 = c31 / r3 + eps
    c32 = c32 / r3 + eps
    c33 = c33 / r3 + eps

    d0 = c00 + c10 + c20 + c30 + eps
    d1 = c01 + c11 + c21 + c31 + eps
    d2 = c02 + c12 + c22 + c32 + eps
    d3 = c03 + c13 + c23 + c33 + eps
    c00 = c00 / d0
    c10 = c10 / d0
    c20 = c20 / d0
    c30 = c30 / d0
    c01 = c01 / d1
    c11 = c11 / d1
    c21 = c21 / d1
    c31 = c31 / d1
    c02 = c02 / d2
    c12 = c12 / d2
    c22 = c22 / d2
    c32 = c32 / d2
    c03 = c03 / d3
    c13 = c13 / d3
    c23 = c23 / d3
    c33 = c33 / d3

    for _ in tl.static_range(0, sinkhorn_iters - 1):
        r0 = c00 + c01 + c02 + c03 + eps
        r1 = c10 + c11 + c12 + c13 + eps
        r2 = c20 + c21 + c22 + c23 + eps
        r3 = c30 + c31 + c32 + c33 + eps
        c00 = c00 / r0
        c01 = c01 / r0
        c02 = c02 / r0
        c03 = c03 / r0
        c10 = c10 / r1
        c11 = c11 / r1
        c12 = c12 / r1
        c13 = c13 / r1
        c20 = c20 / r2
        c21 = c21 / r2
        c22 = c22 / r2
        c23 = c23 / r2
        c30 = c30 / r3
        c31 = c31 / r3
        c32 = c32 / r3
        c33 = c33 / r3

        d0 = c00 + c10 + c20 + c30 + eps
        d1 = c01 + c11 + c21 + c31 + eps
        d2 = c02 + c12 + c22 + c32 + eps
        d3 = c03 + c13 + c23 + c33 + eps
        c00 = c00 / d0
        c10 = c10 / d0
        c20 = c20 / d0
        c30 = c30 / d0
        c01 = c01 / d1
        c11 = c11 / d1
        c21 = c21 / d1
        c31 = c31 / d1
        c02 = c02 / d2
        c12 = c12 / d2
        c22 = c22 / d2
        c32 = c32 / d2
        c03 = c03 / d3
        c13 = c13 / d3
        c23 = c23 / d3
        c33 = c33 / d3

    out = comb + b * stride_cb + s * stride_cs
    tl.store(out + 0 * stride_ci + 0 * stride_cj, c00)
    tl.store(out + 0 * stride_ci + 1 * stride_cj, c01)
    tl.store(out + 0 * stride_ci + 2 * stride_cj, c02)
    tl.store(out + 0 * stride_ci + 3 * stride_cj, c03)
    tl.store(out + 1 * stride_ci + 0 * stride_cj, c10)
    tl.store(out + 1 * stride_ci + 1 * stride_cj, c11)
    tl.store(out + 1 * stride_ci + 2 * stride_cj, c12)
    tl.store(out + 1 * stride_ci + 3 * stride_cj, c13)
    tl.store(out + 2 * stride_ci + 0 * stride_cj, c20)
    tl.store(out + 2 * stride_ci + 1 * stride_cj, c21)
    tl.store(out + 2 * stride_ci + 2 * stride_cj, c22)
    tl.store(out + 2 * stride_ci + 3 * stride_cj, c23)
    tl.store(out + 3 * stride_ci + 0 * stride_cj, c30)
    tl.store(out + 3 * stride_ci + 1 * stride_cj, c31)
    tl.store(out + 3 * stride_ci + 2 * stride_cj, c32)
    tl.store(out + 3 * stride_ci + 3 * stride_cj, c33)


@triton.jit
def _hc_split_sinkhorn4_vector_kernel(
    mixes,
    hc_scale,
    hc_base,
    pre,
    post,
    comb,
    stride_mb: tl.constexpr,
    stride_ms: tl.constexpr,
    stride_pb: tl.constexpr,
    stride_ps: tl.constexpr,
    stride_pv: tl.constexpr,
    stride_ob: tl.constexpr,
    stride_os: tl.constexpr,
    stride_ov: tl.constexpr,
    stride_cb: tl.constexpr,
    stride_cs: tl.constexpr,
    stride_ci: tl.constexpr,
    stride_cj: tl.constexpr,
    seq_len: tl.constexpr,
    eps: tl.constexpr,
    sinkhorn_iters: tl.constexpr,
):
    pid = tl.program_id(0)
    b = pid // seq_len
    s = pid - b * seq_len
    mix = mixes + b * stride_mb + s * stride_ms

    scale0 = tl.load(hc_scale)
    scale1 = tl.load(hc_scale + 1)
    scale2 = tl.load(hc_scale + 2)
    offs4 = tl.arange(0, 4)

    pre_v = tl.sigmoid(
        tl.load(mix + offs4) * scale0 + tl.load(hc_base + offs4)
    ) + eps
    post_v = 2.0 * tl.sigmoid(
        tl.load(mix + 4 + offs4) * scale1 + tl.load(hc_base + 4 + offs4)
    )
    tl.store(pre + b * stride_pb + s * stride_ps + offs4 * stride_pv, pre_v)
    tl.store(post + b * stride_ob + s * stride_os + offs4 * stride_ov, post_v)

    offs16 = tl.arange(0, 16)
    rows = offs16 // 4
    cols = offs16 % 4
    values = (
        tl.load(mix + 8 + offs16) * scale2 + tl.load(hc_base + 8 + offs16)
    )
    matrix = tl.reshape(values, (4, 4))
    matrix = tl.exp(matrix - tl.max(matrix, axis=1)[:, None])
    matrix = matrix / tl.sum(matrix, axis=1)[:, None] + eps
    matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    for _ in tl.static_range(0, sinkhorn_iters - 1):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + eps)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)

    out = comb + b * stride_cb + s * stride_cs
    tl.store(
        out + rows * stride_ci + cols * stride_cj,
        tl.reshape(matrix, (16,)),
    )


def hc_split_sinkhorn4_triton(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    sinkhorn_iters: int,
    eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not torch.version.hip
        or hc_mult != 4
        or mixes.ndim != 3
        or mixes.shape[-1] != 24
        or hc_scale.numel() < 3
        or hc_base.numel() < 24
        or sinkhorn_iters < 1
    ):
        return None

    b, s, _ = mixes.shape
    pre = torch.empty((b, s, 4), dtype=torch.float32, device=mixes.device)
    post = torch.empty((b, s, 4), dtype=torch.float32, device=mixes.device)
    comb = torch.empty((b, s, 4, 4), dtype=torch.float32, device=mixes.device)
    grid = (b * s,)
    _hc_split_sinkhorn4_vector_kernel[grid](
        mixes,
        hc_scale,
        hc_base,
        pre,
        post,
        comb,
        mixes.stride(0),
        mixes.stride(1),
        pre.stride(0),
        pre.stride(1),
        pre.stride(2),
        post.stride(0),
        post.stride(1),
        post.stride(2),
        comb.stride(0),
        comb.stride(1),
        comb.stride(2),
        comb.stride(3),
        seq_len=s,
        eps=eps,
        sinkhorn_iters=sinkhorn_iters,
        num_warps=1,
    )
    return pre, post, comb


@tilelang.jit(pass_configs=pass_configs)
def hc_split_sinkhorn_kernel(hc: int, sinkhorn_iters: int, eps: float):
    n = T.symbolic("n")
    mix_hc = (2 + hc) * hc
    threads = 64

    ENABLE_PDL = is_arch_support_pdl()

    @T.prim_func
    def hc_split_sinkhorn_kernel_(
        mixes: T.Tensor[(n, mix_hc), FP32],
        hc_scale: T.Tensor[(3,), T.float32],
        hc_base: T.Tensor[(mix_hc,), T.float32],
        pre: T.Tensor[(n, hc), FP32],
        post: T.Tensor[(n, hc), FP32],
        comb: T.Tensor[(n, hc, hc), FP32],
    ):
        with T.Kernel(n, threads=threads) as i:
            if ENABLE_PDL:
                T.pdl_sync()

            mixes_shared = T.alloc_shared(mix_hc, FP32)
            comb_frag = T.alloc_fragment((hc, hc), FP32)
            T.copy(mixes[i, :], mixes_shared)

            for j in T.Parallel(hc):
                pre[i, j] = T.sigmoid(mixes_shared[j] * hc_scale[0] + hc_base[j]) + eps
            for j in T.Parallel(hc):
                post[i, j] = 2 * T.sigmoid(
                    mixes_shared[j + hc] * hc_scale[1] + hc_base[j + hc]
                )
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = (
                    mixes_shared[j * hc + k + hc * 2] * hc_scale[2]
                    + hc_base[j * hc + k + hc * 2]
                )

            row_sum = T.alloc_fragment(hc, FP32)
            col_sum = T.alloc_fragment(hc, FP32)

            row_max = T.alloc_fragment(hc, FP32)
            T.reduce_max(comb_frag, row_max, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = T.exp(comb_frag[j, k] - row_max[j])
            T.reduce_sum(comb_frag, row_sum, dim=1)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / row_sum[j] + eps

            T.reduce_sum(comb_frag, col_sum, dim=0)
            for j, k in T.Parallel(hc, hc):
                comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            for _ in T.serial(sinkhorn_iters - 1):
                T.reduce_sum(comb_frag, row_sum, dim=1)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (row_sum[j] + eps)
                T.reduce_sum(comb_frag, col_sum, dim=0)
                for j, k in T.Parallel(hc, hc):
                    comb_frag[j, k] = comb_frag[j, k] / (col_sum[k] + eps)

            T.copy(comb_frag, comb[i, :, :])
            if ENABLE_PDL:
                T.pdl_trigger()

    return hc_split_sinkhorn_kernel_


def hc_split_sinkhorn(
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int = 4,
    sinkhorn_iters: int = 20,
    eps: float = 1e-6,
    gfx90a_global_batch_size: int | None = None,
):
    b, s, _ = mixes.size()

    # TileLang 0.1.11 emits an HSACO that cannot be loaded on gfx90a. Keep the
    # exact reference math available there; hc_mult is only 4 for DSV4, so this
    # is small compared with the surrounding attention and MoE work.
    props = torch.cuda.get_device_properties(mixes.device)
    if getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx90a":
        from sglang.srt.model_executor.runner import get_is_capture_mode

        # The load-time MHC prewarm compiles and validates the native kernel.
        # Avoid enqueuing it in the two discarded eager graph warmups: a native
        # JIT launch on the auxiliary MHC path immediately before Mori's
        # collective warmup can leave ranks at different queue positions. The
        # actual torch.cuda.graph capture still takes the native branch.
        graph_warmup = (
            get_is_capture_mode() and not torch.cuda.is_current_stream_capturing()
        )
        if (
            envs.SGLANG_DSV4_GFX90A_NATIVE_MHC_SINKHORN.get()
            and sinkhorn_iters == 20
        ):
            from sglang.kernels.ops.layernorm.gfx90a_mhc_sinkhorn import (
                gfx90a_mhc_sinkhorn_wave64,
                preload_gfx90a_mhc_sinkhorn,
            )

            # This decision must be rank invariant inside a Mori graph. Local
            # token shards can differ, while ForwardBatch.batch_size cannot.
            native_iters = (
                envs.SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS.get()
                if gfx90a_global_batch_size == 1
                else 20
            )
            if graph_warmup:
                preload_gfx90a_mhc_sinkhorn(native_iters)
            else:
                native_result = gfx90a_mhc_sinkhorn_wave64(
                    mixes, hc_scale, hc_base, eps, native_iters
                )
                if native_result is not None:
                    return native_result
        triton_result = hc_split_sinkhorn4_triton(
            mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters, eps
        )
        if triton_result is not None:
            return triton_result

        mixes = mixes.view(b, s, (2 + hc_mult) * hc_mult)
        pre = (
            torch.sigmoid(mixes[..., :hc_mult] * hc_scale[0] + hc_base[:hc_mult])
            + eps
        )
        post = 2 * torch.sigmoid(
            mixes[..., hc_mult : 2 * hc_mult] * hc_scale[1]
            + hc_base[hc_mult : 2 * hc_mult]
        )
        comb = (
            mixes[..., 2 * hc_mult :] * hc_scale[2] + hc_base[2 * hc_mult :]
        ).view(b, s, hc_mult, hc_mult)
        comb = torch.softmax(comb, dim=-1) + eps
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        for _ in range(sinkhorn_iters - 1):
            comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
            comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
        return pre, post, comb

    pre = mixes.new_empty(b, s, hc_mult)
    post = mixes.new_empty(b, s, hc_mult)
    comb = mixes.new_empty(b, s, hc_mult, hc_mult)
    kernel = hc_split_sinkhorn_kernel(hc_mult, sinkhorn_iters, eps)
    kernel(
        mixes.view(-1, (2 + hc_mult) * hc_mult),
        hc_scale,
        hc_base,
        pre.view(-1, hc_mult),
        post.view(-1, hc_mult),
        comb.view(-1, hc_mult, hc_mult),
    )
    return pre, post, comb


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(512, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )
            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                xs = T.alloc_shared((hc_mult, hidden_block), T.float32)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


@tilelang.jit
def mhc_pre_gemm_sqrsum_tilelang(
    x,
    fn,
    out,
    sqrsum,
    hc_mult3: int,
    hc_hidden_size: int,
    token_block: int = 32,
    hidden_block: int = 256,
):
    assert hc_mult3 <= 32
    num_tokens = T.dynamic("num_tokens")
    assert hc_hidden_size % hidden_block == 0

    x: T.Tensor((num_tokens, hc_hidden_size), T.bfloat16)
    fn: T.Tensor((hc_mult3, hc_hidden_size), T.float32)
    out: T.Tensor((num_tokens, hc_mult3), T.float32)
    sqrsum: T.Tensor((num_tokens), T.float32)

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(T.ceildiv(num_tokens, token_block)) as px:
        out_frag = T.alloc_fragment((token_block, 32), T.float32)
        sqrsum_part = T.alloc_fragment((token_block, 4), T.float32)
        T.clear(out_frag)
        T.clear(sqrsum_part)
        if ENABLE_PDL:
            T.pdl_sync()
        for pz in T.Pipelined(hc_hidden_size // hidden_block, num_stages=2):
            x_smem_16 = T.alloc_shared((token_block, hidden_block), T.bfloat16)
            fn_smem = T.alloc_shared((32, hidden_block), T.float32)

            T.annotate_layout(
                {x_smem_16: tilelang.layout.make_swizzled_layout(x_smem_16)}
            )

            T.copy(x[px * token_block, pz * hidden_block], x_smem_16)
            T.copy(fn[0, pz * hidden_block], fn_smem)

            x_frag_16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
            T.copy(x_smem_16, x_frag_16)
            x_frag = T.alloc_fragment((token_block, hidden_block), T.float32)
            T.copy(x_frag_16, x_frag)

            for jj in T.serial(hidden_block // 4):
                for i, j in T.Parallel(token_block, 4):
                    sqrsum_part[i, j] += x_frag[i, jj * 4 + j] * x_frag[i, jj * 4 + j]

            T.gemm(
                x_frag,
                fn_smem,
                out_frag,
                transpose_A=False,
                transpose_B=True,
                clear_accum=False,
            )
        sqrsum_l = T.alloc_fragment(token_block, T.float32)
        T.reduce_sum(sqrsum_part, sqrsum_l)
        for i in T.Parallel(token_block):
            sqrsum[px * token_block + i] = sqrsum_l[i]
        for i, j in T.Parallel(token_block, 32):
            if j < hc_mult3:
                out[px * token_block + i, j] = out_frag[i, j]
        if ENABLE_PDL:
            T.pdl_trigger()


@functools.cache
def mhc_pre_gemm_sqrsum_splitk_kernel(
    hc_mult3: int,
    hc_hidden_size: int,
    split_k: int,
    token_block: int = 32,
    hidden_block: int = 256,
    threads: int = 128,
):
    _load_tilelang()
    assert hc_mult3 <= 32
    assert hc_hidden_size % hidden_block == 0
    assert hc_hidden_size % split_k == 0
    split_size = hc_hidden_size // split_k
    assert split_size % hidden_block == 0

    num_tokens = T.dynamic("num_tokens")

    ENABLE_PDL = is_arch_support_pdl()

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_splitk_stage_0(
        x: T.Tensor[(num_tokens, hc_hidden_size), T.bfloat16],
        fn: T.Tensor[(hc_mult3, hc_hidden_size), T.float32],
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
    ):
        with T.Kernel(T.ceildiv(num_tokens, token_block), split_k, threads=threads) as (
            px,
            bz,
        ):
            out_frag = T.alloc_fragment((token_block, 32), T.float32)
            sq_part4 = T.alloc_fragment((token_block, 4), T.float32)
            T.clear(out_frag)
            T.clear(sq_part4)

            k_base = bz * split_size

            if ENABLE_PDL:
                T.pdl_sync()

            for pz in T.Pipelined(split_size // hidden_block, num_stages=2):
                x_smem = T.alloc_shared((token_block, hidden_block), T.bfloat16)
                fn_smem = T.alloc_shared((32, hidden_block), T.float32)

                T.annotate_layout(
                    {x_smem: tilelang.layout.make_swizzled_layout(x_smem)}
                )

                T.copy(x[px * token_block, k_base + pz * hidden_block], x_smem)
                T.copy(fn[0, k_base + pz * hidden_block], fn_smem)

                x_f16 = T.alloc_fragment((token_block, hidden_block), T.bfloat16)
                T.copy(x_smem, x_f16)
                x_f = T.alloc_fragment((token_block, hidden_block), T.float32)
                T.copy(x_f16, x_f)

                for jj in T.serial(hidden_block // 4):
                    for i, j in T.Parallel(token_block, 4):
                        v = x_f[i, jj * 4 + j]
                        sq_part4[i, j] += v * v

                T.gemm(
                    x_f,
                    fn_smem,
                    out_frag,
                    transpose_A=False,
                    transpose_B=True,
                    clear_accum=False,
                )

            sq_l = T.alloc_fragment((token_block,), T.float32)
            T.reduce_sum(sq_part4, sq_l)

            for i in T.Parallel(token_block):
                t = px * token_block + i
                if t < num_tokens:
                    sqrsum_partial[bz, t] = sq_l[i]

            for i, j in T.Parallel(token_block, 32):
                t = px * token_block + i
                if t < num_tokens:
                    out_partial[bz, t, j] = out_frag[i, j]

            if ENABLE_PDL:
                T.pdl_trigger()

    @tilelang.jit
    def mhc_pre_gemm_sqrsum_splitk_stage_1(
        out_partial: T.Tensor[(split_k, num_tokens, 32), T.float32],
        sqrsum_partial: T.Tensor[(split_k, num_tokens), T.float32],
        out: T.Tensor[(num_tokens, hc_mult3), T.float32],
        sqrsum: T.Tensor[(num_tokens,), T.float32],
    ):
        warps_per_cta = threads // 32
        num_reduce = T.ceildiv(split_k, 32)
        with T.Kernel(T.ceildiv(num_tokens, warps_per_cta), threads=threads) as (px,):
            tx = T.get_thread_binding()
            warp = tx // 32
            lane = tx % 32
            t = px * warps_per_cta + warp
            s = T.alloc_local((1,), T.float32)
            acc = T.alloc_local((1,), T.float32)
            s[0] = 0
            acc[0] = 0
            if ENABLE_PDL:
                T.pdl_sync()

            if t < num_tokens:
                for r in T.serial(num_reduce):
                    bz = r * 32 + lane
                    s[0] += T.if_then_else(bz < split_k, sqrsum_partial[bz, t], 0.0)
                sqrsum[t] = T.warp_reduce_sum(s[0])
                if lane < hc_mult3:
                    for bz in T.serial(split_k):
                        acc[0] += out_partial[bz, t, lane]
                    out[t, lane] = acc[0]

            if ENABLE_PDL:
                T.pdl_trigger()

    return (
        mhc_pre_gemm_sqrsum_splitk_stage_0,
        mhc_pre_gemm_sqrsum_splitk_stage_1,
    )


def _compute_num_split_for_mhc_pre(num_tokens: int, hc_hidden_size: int) -> int:
    block_m, block_k = 64, 64
    grid_size = (num_tokens + block_m - 1) // block_m
    num_block_k = (hc_hidden_size + block_k - 1) // block_k

    n_sms = torch.cuda.get_device_properties(0).multi_processor_count

    return max(1, min(n_sms // max(grid_size, 1), num_block_k // 4))


def get_mhc_pre_token_count_representatives(
    max_num_tokens: int, hc_hidden_size: int
) -> Tuple[int, ...]:
    """One representative token count per distinct mhc_pre n_splits bucket over
    [1, max_num_tokens] (the kernel is specialized only by n_splits)."""
    reps = {}
    for grid in range(1, (max(1, max_num_tokens) + 63) // 64 + 1):
        num_tokens = min(grid * 64, max_num_tokens)
        reps[_compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)] = num_tokens
    return tuple(sorted(reps.values()))


def prewarm_mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int,
    n_splits_pre: int,
    norm_weight: torch.Tensor | None,
    norm_eps: float | None,
):
    """Compile the prenorm kernel for every n_splits bucket by replaying the
    prenorm with the call's real weights. The compiled kernels are written to
    the TileLang/DeepGEMM on-disk JIT cache, so this cost is paid only on a cold
    cache; later server runs hit the cache. Driven once per process from load_weights.
    """
    from sglang.srt.runtime_context import get_schedule

    hc_mult, hidden_size = residual.shape[-2], residual.shape[-1]
    max_num_tokens = max(1, get_schedule().chunked_prefill_size)
    buckets = get_mhc_pre_token_count_representatives(
        max_num_tokens, hc_mult * hidden_size
    )

    logger.info("DeepSeek V4 MHC prenorm prewarm: %d n_splits buckets", len(buckets))
    with torch.inference_mode():
        for num_tokens in buckets:
            mhc_pre(
                residual.new_zeros(num_tokens, hc_mult, hidden_size),
                fn,
                hc_scale,
                hc_base,
                rms_eps,
                hc_pre_eps,
                hc_sinkhorn_eps,
                hc_post_mult_value,
                sinkhorn_repeat,
                n_splits,
                n_splits_pre,
                norm_weight=norm_weight,
                norm_eps=norm_eps,
            )


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_pre_big_fuse_with_norm_tilelang(
    gemm_out_mul,
    gemm_out_sqrsum,
    hc_scale,
    hc_base,
    residual,
    post_mix,
    comb_mix,
    layer_input,
    norm_weight,
    hidden_size: int,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    norm_eps: float,
    n_splits: int = 16,
    hc_mult: int = 4,
    gemm_last_dim: int = -1,
):
    """Fused mhc_pre big_fuse + RMSNorm of layer_input.

    Identical to mhc_pre_big_fuse_tilelang for the (post_mix, comb_mix) path.
    For the layer_input path, the weighted-sum result is stashed in shared
    memory while accumulating sum_sq, then a second pipelined sweep applies
    rsqrt(sum_sq/D + norm_eps) * norm_weight before writing to HBM.
    """
    num_tokens = T.dynamic("num_tokens")
    hc_mult3 = hc_mult * (2 + hc_mult)
    if gemm_last_dim < 0:
        gemm_last_dim = hc_mult3
    hidden_block = math.gcd(1024, hidden_size)

    gemm_out_mul: T.Tensor[[n_splits, num_tokens, gemm_last_dim], T.float32]
    gemm_out_sqrsum: T.Tensor[[n_splits, num_tokens], T.float32]
    hc_scale: T.Tensor[[3], T.float32]
    hc_base: T.Tensor[[hc_mult3], T.float32]
    residual: T.Tensor[[num_tokens, hc_mult, hidden_size], T.bfloat16]
    post_mix: T.Tensor[[num_tokens, hc_mult], T.float32]
    comb_mix: T.Tensor[[num_tokens, hc_mult * hc_mult], T.float32]
    layer_input: T.Tensor[[num_tokens, hidden_size], T.bfloat16]
    norm_weight: T.Tensor[[hidden_size], T.bfloat16]

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(num_tokens, threads=96) as i:
        rms = T.alloc_fragment(1, T.float32)
        mixes = T.alloc_fragment(hc_mult3, T.float32)
        T.clear(mixes)
        rms[0] = 0

        if ENABLE_PDL:
            T.pdl_sync()

        for i_split in T.serial(n_splits):
            rms[0] += gemm_out_sqrsum[i_split, i]
        rms[0] = T.rsqrt(rms[0] / (hc_mult * hidden_size) + rms_eps)
        for j in T.Parallel(hc_mult3):
            mixes[j] = 0
            for i_split in T.serial(n_splits):
                mixes[j] += gemm_out_mul[i_split, i, j]
            mixes[j] *= rms[0]
        mixes_shared = T.alloc_shared(hc_mult3, T.float32)
        T.copy(mixes, mixes_shared)

        if T.get_thread_binding() < 32:
            cm = T.alloc_fragment((hc_mult, hc_mult), T.float32)
            for j in T.Parallel(hc_mult):
                post_mix[i, j] = (
                    T.sigmoid(
                        mixes_shared[j + hc_mult] * hc_scale[1] + hc_base[j + hc_mult]
                    )
                    * hc_post_mult_value
                )
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = (
                    mixes_shared[j * hc_mult + k + hc_mult * 2] * hc_scale[2]
                    + hc_base[j * hc_mult + k + hc_mult * 2]
                )

            row_sum = T.alloc_fragment(hc_mult, T.float32)
            col_sum = T.alloc_fragment(hc_mult, T.float32)

            row_max = T.alloc_fragment(hc_mult, T.float32)
            T.reduce_max(cm, row_max, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = T.exp(cm[j, k] - row_max[j])
            T.reduce_sum(cm, row_sum, dim=1)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / row_sum[j] + hc_sinkhorn_eps

            T.reduce_sum(cm, col_sum, dim=0)
            for j, k in T.Parallel(hc_mult, hc_mult):
                cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for _ in T.serial(sinkhorn_repeat - 1):
                T.reduce_sum(cm, row_sum, dim=1)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (row_sum[j] + hc_sinkhorn_eps)

                T.reduce_sum(cm, col_sum, dim=0)
                for j, k in T.Parallel(hc_mult, hc_mult):
                    cm[j, k] = cm[j, k] / (col_sum[k] + hc_sinkhorn_eps)

            for j, k in T.Parallel(hc_mult, hc_mult):
                comb_mix[i, j * hc_mult + k] = cm[j, k]
        else:
            pre_mix_shared = T.alloc_shared(hc_mult, T.float32)
            for j in T.Parallel(hc_mult):
                pre_mix_shared[j] = (
                    T.sigmoid(
                        mixes_shared[j] * hc_scale[0] + hc_base[j],
                    )
                    + hc_pre_eps
                )

            # Stash unnormalized weighted-sum output in shared memory as bf16
            # (matches the rounding the reference path does when RMSNorm reads bf16).
            output_shared = T.alloc_shared(hidden_size, T.bfloat16)
            sumsq_per_pos = T.alloc_fragment(hidden_block, T.float32)
            T.clear(sumsq_per_pos)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=3):
                xs = T.alloc_shared((hc_mult, hidden_block), T.bfloat16)
                xl = T.alloc_fragment((hc_mult, hidden_block), T.float32)
                T.copy(residual[i, 0, i0_h * hidden_block], xs)
                T.copy(xs, xl)

                ol = T.alloc_fragment(hidden_block, T.float32)
                T.clear(ol)

                for i_hc in T.serial(hc_mult):
                    pre = pre_mix_shared[i_hc]
                    for i1_h in T.Parallel(hidden_block):
                        ol[i1_h] += pre * xl[i_hc, i1_h]

                for i1_h in T.Parallel(hidden_block):
                    sumsq_per_pos[i1_h] += ol[i1_h] * ol[i1_h]
                    output_shared[i0_h * hidden_block + i1_h] = T.bfloat16(ol[i1_h])

            sumsq = T.alloc_fragment(1, T.float32)
            T.reduce_sum(sumsq_per_pos, sumsq, dim=0)
            rsqrt_norm = T.alloc_fragment(1, T.float32)
            rsqrt_norm[0] = T.rsqrt(sumsq[0] / hidden_size + norm_eps)

            for i0_h in T.Pipelined(hidden_size // hidden_block, num_stages=2):
                w_shared = T.alloc_shared(hidden_block, T.bfloat16)
                w_local = T.alloc_fragment(hidden_block, T.float32)
                T.copy(norm_weight[i0_h * hidden_block], w_shared)
                T.copy(w_shared, w_local)

                ol = T.alloc_fragment(hidden_block, T.float32)
                for i1_h in T.Parallel(hidden_block):
                    ol[i1_h] = (
                        output_shared[i0_h * hidden_block + i1_h]
                        * rsqrt_norm[0]
                        * w_local[i1_h]
                    )

                T.copy(ol, layer_input[i, i0_h * hidden_block])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_pre(
    residual: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    n_splits_pre: int = 32,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2

    hc_hidden_size = hc_mult * hidden_size
    assert fn.shape[0] == hc_mult3
    assert fn.shape[1] == hc_hidden_size
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(
        num_tokens, hc_mult, dtype=torch.float32, device=residual.device
    )
    comb_mix = torch.empty(
        num_tokens, hc_mult2, dtype=torch.float32, device=residual.device
    )
    # layer_input is the post-norm activation fed into the MoE. Allocate it in
    # the symmetric memory pool so the downstream all-reduce uses the low-latency
    # NCCL symmetric path: the Triton inplace MoE runner writes the expert
    # output back into this buffer, so a symmetric input yields a symmetric
    # all-reduce input.
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        layer_input = torch.empty(
            num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device
        )

    if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)

        gemm_out_mul = torch.empty(
            n_splits, num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
        )
        gemm_out_sqrsum = torch.empty(
            n_splits, num_tokens, dtype=torch.float32, device=residual.device
        )

        from sglang.srt.layers.deep_gemm_wrapper.entrypoint import tf32_hc_prenorm_gemm

        tf32_hc_prenorm_gemm(
            residual_flat.view(num_tokens, hc_hidden_size),
            fn_flat,
            gemm_out_mul,
            gemm_out_sqrsum,
            n_splits,
        )
        gemm_last_dim = hc_mult3
        big_fuse_n_splits = n_splits
    else:
        if num_tokens <= 2048:
            assert n_splits == 1
            if hc_hidden_size == 16384:
                hidden_block = 256
            elif hc_hidden_size == 28672:
                hidden_block = 128
            else:
                raise NotImplementedError(
                    f"mhc_pre splitk kernel only supports hc_hidden_size in {{16384, 28672}}, "
                    f"got {hc_hidden_size}"
                )
            kernel_0, _ = mhc_pre_gemm_sqrsum_splitk_kernel(
                hc_mult3,
                hc_hidden_size,
                split_k=n_splits_pre,
                token_block=32,
                hidden_block=hidden_block,
            )
            partial_out = torch.empty(
                n_splits_pre,
                num_tokens,
                32,
                dtype=torch.float32,
                device=residual.device,
            )
            partial_sqrsum = torch.empty(
                n_splits_pre, num_tokens, dtype=torch.float32, device=residual.device
            )
            kernel_0(
                residual_flat.view(num_tokens, hc_hidden_size),
                fn_flat,
                partial_out,
                partial_sqrsum,
            )
            # Stage_1 reduction is folded into big_fuse below; skip launching it.
            gemm_out_mul = partial_out
            gemm_out_sqrsum = partial_sqrsum
            gemm_last_dim = 32
            big_fuse_n_splits = n_splits_pre
        else:
            gemm_out_mul = torch.empty(
                n_splits,
                num_tokens,
                hc_mult3,
                dtype=torch.float32,
                device=residual.device,
            )
            gemm_out_sqrsum = torch.empty(
                n_splits, num_tokens, dtype=torch.float32, device=residual.device
            )
            assert (
                n_splits == 1
            ), "The simple TileLang version gemm_sqrsum doesn't support split-k"
            mhc_pre_gemm_sqrsum_tilelang(
                residual_flat.view(num_tokens, hc_mult * hidden_size),
                fn_flat,
                gemm_out_mul.squeeze(0),
                gemm_out_sqrsum.squeeze(0),
                hc_mult3,
                hc_mult * hidden_size,
            )
            gemm_last_dim = hc_mult3
            big_fuse_n_splits = n_splits

    if norm_weight is not None:
        assert norm_eps is not None, "norm_eps required when norm_weight is provided"
        assert norm_weight.shape == (
            hidden_size,
        ), f"norm_weight shape {tuple(norm_weight.shape)} != (hidden_size={hidden_size},)"
        norm_weight_bf = (
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )
    else:
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_flat,
            post_mix,
            comb_mix,
            layer_input,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            big_fuse_n_splits,
            hc_mult,
            gemm_last_dim,
        )

    post_mix = post_mix.view(*outer_shape, hc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, hc_mult, hc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_post_tilelang(
    a, b, c, d, x, hc: int, hidden: int, n_thr: int = 128, h_blk: int = 1024
):
    n = T.dynamic("num_tokens")
    h = hidden

    h_blk = math.gcd(hidden, h_blk)
    a: T.Tensor((n, hc, hc), T.float32)
    b: T.Tensor((n, hc, h), T.bfloat16)
    c: T.Tensor((n, hc), T.float32)
    d: T.Tensor((n, h), T.bfloat16)
    x: T.Tensor((n, hc, h), T.bfloat16)

    ENABLE_PDL = is_arch_support_pdl()
    with T.Kernel(n, threads=n_thr) as i_n:
        if ENABLE_PDL:
            T.pdl_sync()

        x_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        b_shared = T.alloc_shared((hc, h_blk), T.bfloat16)
        d_shared = T.alloc_shared(h_blk, T.bfloat16)

        x_local = T.alloc_fragment((hc, h_blk), T.float32)
        b_local = T.alloc_fragment((hc, h_blk), T.float32)
        d_local = T.alloc_fragment(h_blk, T.float32)

        a_local = T.alloc_fragment((hc, hc), T.float32)
        c_local = T.alloc_fragment(hc, T.float32)
        T.copy(a[i_n, 0, 0], a_local)
        T.copy(c[i_n, 0], c_local)

        for i0_h in T.Pipelined(T.ceildiv(h, h_blk), num_stages=2):
            T.copy(b[i_n, 0, i0_h * h_blk], b_shared)
            T.copy(d[i_n, i0_h * h_blk], d_shared)

            T.copy(b_shared, b_local)
            T.copy(d_shared, d_local)
            for i_hco, i1_h in T.Parallel(hc, h_blk):
                x_local[i_hco, i1_h] = c_local[i_hco] * d_local[i1_h]
                for i_hci in T.serial(hc):
                    x_local[i_hco, i1_h] += a_local[i_hci, i_hco] * b_local[i_hci, i1_h]
            T.copy(x_local, x_shared)

            T.copy(x_shared, x[i_n, 0, i0_h * h_blk])

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_post(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
) -> torch.Tensor:
    if is_dsa_prefill_cp_round_robin_split():
        x = strict_contiguous(x)
        residual = strict_contiguous(residual)
        post_layer_mix = strict_contiguous(post_layer_mix)
        comb_res_mix = strict_contiguous(comb_res_mix)
    out = torch.empty_like(residual)
    mhc_post_tilelang(
        comb_res_mix,
        residual,
        post_layer_mix.squeeze(-1),
        x,
        out,
        residual.shape[-2],
        residual.shape[-1],
    )
    return out


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
        tilelang.PassConfigKey.TL_PTXAS_REGISTER_USAGE_LEVEL: 10,
    },
)
def mhc_fused_post_pre_fma_tilelang(
    prev_comb_mix,
    prev_residual,
    prev_post_mix,
    hidden_in,
    pre_fn,
    mixes_partial_out,
    sqrsum_partial_out,
    cur_residual_out,
    hc: int,
    hidden_size: int,
    num_mix_outputs: int,
    n_thr: int = 256,
    tile_mix_outputs: int = 1,
    split_k: int = 1,
):
    num_tokens = T.dynamic("num_tokens")
    split_k = T.dynamic("split_k")

    hidden_per_split = (hidden_size + split_k - 1) // split_k
    num_mix_output_tiles = (num_mix_outputs + tile_mix_outputs - 1) // tile_mix_outputs

    prev_comb_mix: T.Tensor((num_tokens, hc, hc), T.float32)
    prev_residual: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)
    prev_post_mix: T.Tensor((num_tokens, hc), T.float32)
    hidden_in: T.Tensor((num_tokens, hidden_size), T.bfloat16)
    pre_fn: T.Tensor((num_mix_outputs, hc, hidden_size), T.float32)

    mixes_partial_out: T.Tensor((split_k, num_tokens, num_mix_outputs), T.float32)
    sqrsum_partial_out: T.Tensor((split_k, num_tokens), T.float32)
    cur_residual_out: T.Tensor((num_tokens, hc, hidden_size), T.bfloat16)

    hidden_iters_per_thread = (hidden_per_split + n_thr - 1) // n_thr
    # CDNA uses wave64.  TileLang's get_lane_idx helper currently only lowers
    # in the CUDA codegen, and treating a 256-thread HIP CTA as eight wave32s
    # also over-allocates/reads the cross-wave partial buffer.  Derive both
    # lane and wave ids from the portable thread binding instead.
    warp_size = 64 if torch.version.hip else 32
    num_warps = n_thr // warp_size

    ENABLE_PDL = is_arch_support_pdl()

    # CTA assignment:
    #   token_idx           : this CTA handles one token.
    #   mix_output_tile_idx : this CTA handles a small tile of mix output columns.
    #                          For HC=4, num_mix_outputs = 24:
    #                            [0:4]   -> pre logits
    #                            [4:8]   -> post logits
    #                            [8:24]  -> comb logits
    #   hidden_split_idx    : this CTA handles one split of the hidden dimension.
    #
    # Thread assignment inside one CTA:
    #   Each thread owns several hidden positions in this hidden split:
    #     hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx
    #
    # For each owned hidden_idx, the thread computes:
    #   1. post result: cur_residual[token, :, hidden_idx]
    #   2. sqrsum partial for pre RMS
    #   3. GEMM partial for several mix output columns
    with T.Kernel(
        num_tokens,
        num_mix_output_tiles,
        split_k,
        threads=n_thr,
    ) as (token_idx, mix_output_tile_idx, hidden_split_idx):
        thread_idx = T.get_thread_binding()
        warp_idx = thread_idx // warp_size
        lane_idx = thread_idx % warp_size

        warp_partials = T.alloc_shared((num_warps, tile_mix_outputs + 1), T.float32)
        post_mix_smem = T.alloc_shared((hc,), T.float32)
        comb_mix_smem = T.alloc_shared((hc, hc), T.float32)

        post_mix_for_token = T.alloc_local((hc,), T.float32)
        comb_mix_for_token = T.alloc_local((hc, hc), T.float32)

        mix_acc = T.alloc_local((tile_mix_outputs,), T.float32)
        sqrsum_acc = T.alloc_local((1,), T.float32)
        cur_residual_values = T.alloc_local((hc,), T.float32)

        T.clear(mix_acc)
        T.clear(sqrsum_acc)

        hidden_split_start = hidden_split_idx * hidden_per_split

        if ENABLE_PDL:
            T.pdl_sync()

        # Load post/comb coefficients for this token.
        #
        # PyTorch equivalent:
        #   post = prev_post_mix[token_idx]      # [HC]
        #   comb = prev_comb_mix[token_idx]      # [HC, HC]
        T.copy(prev_post_mix[token_idx, 0], post_mix_smem)
        T.copy(prev_comb_mix[token_idx, 0, 0], comb_mix_smem)

        for route_idx in T.unroll(hc):
            post_mix_for_token[route_idx] = post_mix_smem[route_idx]

        for old_route_idx in T.unroll(hc):
            for new_route_idx in T.unroll(hc):
                comb_mix_for_token[old_route_idx, new_route_idx] = comb_mix_smem[
                    old_route_idx, new_route_idx
                ]

        for hidden_iter in T.serial(hidden_iters_per_thread):
            hidden_idx = hidden_split_start + hidden_iter * n_thr + thread_idx

            if hidden_idx < hidden_size:
                # Step A: fused post.
                #
                # PyTorch equivalent:
                #   cur_residual =
                #       post.unsqueeze(-1) * hidden_in.unsqueeze(1)
                #       + (
                #           comb.unsqueeze(-1)
                #           * prev_residual.unsqueeze(2)
                #         ).sum(dim=1)
                #
                # Scalar form for this token and this hidden position:
                #   cur_residual[j, h]
                #     = post[j] * hidden_in[h]
                #     + sum_k comb[k, j] * prev_residual[k, h]
                for new_route_idx in T.unroll(hc):
                    cur_residual_values[new_route_idx] = (
                        post_mix_for_token[new_route_idx]
                        * hidden_in[token_idx, hidden_idx]
                    )

                    for old_route_idx in T.unroll(hc):
                        cur_residual_values[new_route_idx] += (
                            comb_mix_for_token[old_route_idx, new_route_idx]
                            * prev_residual[token_idx, old_route_idx, hidden_idx]
                        )

                # Match the unfused path:
                #   mhc_post writes bf16 residual,
                #   then mhc_pre reads bf16 residual.
                for route_idx in T.unroll(hc):
                    cur_residual_values[route_idx] = T.bfloat16(
                        cur_residual_values[route_idx]
                    )

                # Step B1: pre sqrsum partial.
                #
                # PyTorch equivalent:
                #   x_flat = cur_residual.reshape(T, HC * H).float()
                #   sqrsum = (x_flat * x_flat).sum(dim=-1)
                #
                # Only mix_output_tile_idx == 0 writes cur_residual and sqrsum,
                # otherwise different output-column CTAs would duplicate this work.
                if mix_output_tile_idx == 0:
                    for route_idx in T.unroll(hc):
                        cur_residual_out[token_idx, route_idx, hidden_idx] = (
                            cur_residual_values[route_idx]
                        )
                        sqrsum_acc[0] += (
                            cur_residual_values[route_idx]
                            * cur_residual_values[route_idx]
                        )

                # Step B2: pre GEMM partial.
                #
                # PyTorch equivalent:
                #   mixes = F.linear(x_flat, fn)
                #
                # Scalar form:
                #   mixes[token, o] +=
                #       pre_fn[o, route, hidden] * cur_residual[route, hidden]
                #
                # This CTA computes only tile_mix_outputs columns of mixes.
                for tile_col_idx in T.unroll(tile_mix_outputs):
                    mix_output_idx = (
                        mix_output_tile_idx * tile_mix_outputs + tile_col_idx
                    )

                    if mix_output_idx < num_mix_outputs:
                        for route_idx in T.unroll(hc):
                            mix_acc[tile_col_idx] += (
                                pre_fn[mix_output_idx, route_idx, hidden_idx]
                                * cur_residual_values[route_idx]
                            )

        # Reduce thread partials inside each warp.
        for tile_col_idx in T.unroll(tile_mix_outputs):
            mix_acc[tile_col_idx] = T.warp_reduce_sum(mix_acc[tile_col_idx])

        if mix_output_tile_idx == 0:
            sqrsum_acc[0] = T.warp_reduce_sum(sqrsum_acc[0])

        # One lane per warp writes warp-level partials to shared memory.
        if lane_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                warp_partials[warp_idx, tile_col_idx] = mix_acc[tile_col_idx]

            if mix_output_tile_idx == 0:
                warp_partials[warp_idx, tile_mix_outputs] = sqrsum_acc[0]

        T.sync_threads()

        # Reduce across warps and write split partials.
        #
        # The full PyTorch result would be:
        #   mixes = F.linear(cur_residual.reshape(T, HC * H), fn)
        #   sqrsum = (cur_residual.float() ** 2).sum(dim=(1, 2))
        #
        # This kernel is split along hidden, so each CTA writes only:
        #   mixes_partial_out[hidden_split_idx, token, o]
        #   sqrsum_partial_out[hidden_split_idx, token]
        #
        # Later mhc_pre_big_fuse does:
        #   mixes = mixes_partial_out.sum(dim=0)
        #   sqrsum = sqrsum_partial_out.sum(dim=0)
        #   rms = rsqrt(sqrsum / (HC * H) + eps)
        #   mixes *= rms
        #   mixes -> pre/post/comb
        #   layer_input = sum_j pre[j] * cur_residual[j]
        if warp_idx == 0:
            for tile_col_idx in T.unroll(tile_mix_outputs):
                mix_output_idx = mix_output_tile_idx * tile_mix_outputs + tile_col_idx

                if mix_output_idx < num_mix_outputs and lane_idx == tile_col_idx:
                    mix_output_partial = T.alloc_var(T.float32, init=0.0)

                    for reduce_warp_idx in T.unroll(num_warps):
                        mix_output_partial += warp_partials[
                            reduce_warp_idx, tile_col_idx
                        ]

                    mixes_partial_out[hidden_split_idx, token_idx, mix_output_idx] = (
                        mix_output_partial
                    )

            if mix_output_tile_idx == 0 and lane_idx == 0:
                sqrsum_partial = T.alloc_var(T.float32, init=0.0)

                for reduce_warp_idx in T.unroll(num_warps):
                    sqrsum_partial += warp_partials[reduce_warp_idx, tile_mix_outputs]

                sqrsum_partial_out[hidden_split_idx, token_idx] = sqrsum_partial

        if ENABLE_PDL:
            T.pdl_trigger()


def mhc_fused_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    post_layer_mix: torch.Tensor,
    comb_res_mix: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    rms_eps: float,
    hc_pre_eps: float,
    hc_sinkhorn_eps: float,
    hc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 1,
    tile_n: int = 1,
    *,
    norm_weight: torch.Tensor | None = None,
    norm_eps: float | None = None,
    global_batch_size: int | None = None,
    fn_bf16: torch.Tensor | None = None,
    fn_fp16: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse the boundary between one mHC post step and the next mHC pre step.

    The unfused sequence is ``mhc_post -> pre-norm GEMM -> mhc_pre big_fuse``.
    This wrapper keeps the numerically sensitive ``mhc_pre_big_fuse`` stage,
    including optional RMSNorm, but removes the separate post/pre boundary.
    Small token batches use the FMA kernel above to combine ``mhc_post`` and the
    pre-norm GEMM in one launch; larger batches keep DeepGEMM for throughput and
    only fuse the Python/model-level scheduling boundary.

    Returns:
        residual_cur: post-mapped residual, shape (..., hc_mult, hidden_size)
        post_mix_cur: shape (..., hc_mult, 1)
        comb_mix_cur: shape (..., hc_mult, hc_mult)
        layer_input_cur: shape (..., hidden_size)
    """

    assert residual.dtype == torch.bfloat16
    assert x.dtype == torch.bfloat16
    assert post_layer_mix.dtype == torch.float32
    assert comb_res_mix.dtype == torch.float32
    assert fn.dtype == torch.float32
    assert hc_scale.dtype == torch.float32
    assert hc_base.dtype == torch.float32

    hc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    hc_mult2 = hc_mult * hc_mult
    hc_mult3 = hc_mult * 2 + hc_mult2
    hc_hidden_size = hc_mult * hidden_size
    outer_shape = residual.shape[:-2]

    assert x.shape == (*outer_shape, hidden_size)
    assert post_layer_mix.shape in (
        (*outer_shape, hc_mult, 1),
        (*outer_shape, hc_mult),
    )
    assert comb_res_mix.shape == (*outer_shape, hc_mult, hc_mult)
    assert fn.shape == (hc_mult3, hc_hidden_size)
    assert hc_scale.shape == (3,)
    assert hc_base.shape == (hc_mult3,)

    residual_flat = residual.view(-1, hc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    if num_tokens == 0:
        # Some DP/EP ranks can receive no tokens; return correctly typed empty
        # tensors so later fused layers keep the same contracts as mhc_pre/hc_post.
        return (
            torch.empty_like(residual),
            torch.empty(
                (*outer_shape, hc_mult, 1), dtype=torch.float32, device=residual.device
            ),
            torch.empty(
                (*outer_shape, hc_mult, hc_mult),
                dtype=torch.float32,
                device=residual.device,
            ),
            torch.empty(
                (*outer_shape, hidden_size),
                dtype=torch.bfloat16,
                device=residual.device,
            ),
        )
    x_flat = x.view(num_tokens, hidden_size)

    # TileLang's fused MFMA path is not a valid gfx90a fallback: its small-token
    # kernel can poison a following Mori graph, while the large-token GEMM does
    # not lower on CDNA2 (notably for a 256-token prefill arriving beside a
    # decode batch).  The Triton/native decomposition below is shape-general,
    # so use it for both decode and prefill on gfx90a.
    props = torch.cuda.get_device_properties(residual.device)
    if (
        getattr(props, "gcnArchName", "").split(":", 1)[0] == "gfx90a"
        and hc_mult == 4
        and hidden_size == 4096
        and sinkhorn_repeat == 20
    ):
        if (
            envs.SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE_FULL.get()
            # Use a rank-invariant graph-tier predicate.  ``num_tokens`` is a
            # local Mori/attention-TP shard and can differ across ranks; using
            # it here makes ranks capture different kernel sequences and the
            # following collective spins.  ForwardBatch.batch_size is global
            # within this TP/EP group and therefore safe for graph capture.
            and global_batch_size == 1
            and fn_fp16 is not None
            and norm_weight is not None
            and norm_eps is not None
        ):
            from sglang.kernels.ops.layernorm.gfx90a_mhc_post_pre import (
                gfx90a_mhc_post_pre,
                preload_gfx90a_mhc_post_pre,
            )
            from sglang.srt.model_executor.runner import get_is_capture_mode

            graph_warmup = (
                get_is_capture_mode()
                and not torch.cuda.is_current_stream_capturing()
            )
            # Compile during the discarded eager warmup, but do not launch a
            # native JIT kernel immediately before Mori's warmup collectives.
            preload_gfx90a_mhc_post_pre()
            if not graph_warmup:
                norm_weight_bf = (
                    norm_weight.bfloat16()
                    if norm_weight.dtype != torch.bfloat16
                    else norm_weight
                ).contiguous()
                with use_symmetric_memory(
                    get_tp_group(), disabled=not is_allocation_symmetric()
                ):
                    native = gfx90a_mhc_post_pre(
                        x_flat,
                        residual_flat,
                        post_layer_mix.view(num_tokens, hc_mult),
                        comb_res_mix.view(num_tokens, hc_mult, hc_mult),
                        fn_fp16,
                        hc_scale,
                        hc_base,
                        norm_weight_bf,
                        rms_eps,
                        hc_sinkhorn_eps,
                        hc_post_mult_value,
                        norm_eps,
                    )
                if native is not None:
                    residual_cur, post_cur, comb_cur, layer_input = native
                    return (
                        residual_cur.view(*outer_shape, hc_mult, hidden_size),
                        post_cur.view(*outer_shape, hc_mult),
                        comb_cur.view(*outer_shape, hc_mult, hc_mult),
                        layer_input.view(*outer_shape, hidden_size),
                    )

        rms_partials = None
        if envs.SGLANG_DSV4_GFX90A_FUSE_MHC_POST_RMS_PARTIALS.get():
            post_result = mhc_post_combine_rms_triton(
                x_flat,
                residual_flat,
                post_layer_mix.view(num_tokens, hc_mult),
                comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            )
            if post_result is None:
                residual_cur = None
            else:
                residual_cur, rms_partials = post_result
        else:
            residual_cur = mhc_post_combine_triton(
                x_flat,
                residual_flat,
                post_layer_mix.view(num_tokens, hc_mult),
                comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            )
        if residual_cur is None:
            raise RuntimeError("gfx90a fused MHC post-combine rejected its shape")
        if (
            envs.SGLANG_DSV4_GFX90A_FUSED_MHC_SPLITK_TAIL.get()
            and rms_partials is not None
            and norm_weight is not None
            and norm_eps is not None
            and global_batch_size == 1
        ):
            norm_weight_bf = (
                norm_weight.bfloat16()
                if norm_weight.dtype != torch.bfloat16
                else norm_weight
            ).contiguous()
            fused_tail = gfx90a_mhc_splitk_fused_tail_triton(
                residual_cur,
                fn,
                fn_fp16,
                rms_partials,
                hc_scale,
                hc_base,
                norm_weight_bf,
                hc_sinkhorn_eps,
                norm_eps,
                global_batch_size,
            )
            if fused_tail is not None:
                post_cur, comb_cur, layer_input = fused_tail
                return (
                    residual_cur.view(*outer_shape, hc_mult, hidden_size),
                    post_cur.view(*outer_shape, hc_mult),
                    comb_cur.view(*outer_shape, hc_mult, hc_mult),
                    layer_input.view(*outer_shape, hidden_size),
                )
        use_bf16_mix = (
            envs.SGLANG_DSV4_GFX90A_BF16_MHC_DOT.get()
            and fn_bf16 is not None
        )
        if use_bf16_mix:
            from sglang.srt.model_executor.runner import get_is_capture_mode

            use_bf16_mix = not (
                get_is_capture_mode()
                and not torch.cuda.is_current_stream_capturing()
            )
        if use_bf16_mix:
            mixes = gfx90a_mhc_pre_mix_bf16_dot_triton(
                residual_cur, fn_bf16, rms_eps
            )
        elif rms_partials is not None:
            if envs.SGLANG_DSV4_GFX90A_SPLITK_MHC_PRE_MIX.get():
                mixes = gfx90a_mhc_pre_mix_splitk_from_partials_triton(
                    residual_cur,
                    fn,
                    rms_partials,
                    rms_eps,
                    global_batch_size,
                )
            if mixes is None:
                mixes = gfx90a_mhc_pre_mix_from_partials_triton(
                    residual_cur, fn, rms_partials, rms_eps
                )
        else:
            mixes = gfx90a_mhc_pre_mix_triton(residual_cur, fn, rms_eps)
        if mixes is None:
            raise RuntimeError("gfx90a fused MHC pre-mix rejected its shape")
        if (
            envs.SGLANG_DSV4_GFX90A_NATIVE_MHC_POST_PRE.get()
            and global_batch_size == 1
            and norm_weight is not None
            and norm_eps is not None
        ):
            from sglang.kernels.ops.layernorm.gfx90a_mhc_post_pre import (
                gfx90a_mhc_finish,
                preload_gfx90a_mhc_post_pre,
            )
            from sglang.srt.model_executor.runner import get_is_capture_mode

            graph_warmup = (
                get_is_capture_mode()
                and not torch.cuda.is_current_stream_capturing()
            )
            preload_gfx90a_mhc_post_pre()
            if not graph_warmup:
                norm_weight_bf = (
                    norm_weight.bfloat16()
                    if norm_weight.dtype != torch.bfloat16
                    else norm_weight
                ).contiguous()
                with use_symmetric_memory(
                    get_tp_group(), disabled=not is_allocation_symmetric()
                ):
                    native_tail = gfx90a_mhc_finish(
                        residual_cur,
                        mixes.view(num_tokens, 24),
                        hc_scale,
                        hc_base,
                        norm_weight_bf,
                        hc_sinkhorn_eps,
                        hc_post_mult_value,
                        norm_eps,
                    )
                if native_tail is not None:
                    post_cur, comb_cur, layer_input = native_tail
                    return (
                        residual_cur.view(*outer_shape, hc_mult, hidden_size),
                        post_cur.view(*outer_shape, hc_mult),
                        comb_cur.view(*outer_shape, hc_mult, hc_mult),
                        layer_input.view(*outer_shape, hidden_size),
                    )
        pre_cur, post_cur, comb_cur = hc_split_sinkhorn(
            mixes,
            hc_scale,
            hc_base,
            hc_mult,
            sinkhorn_repeat,
            hc_sinkhorn_eps,
            global_batch_size,
        )
        with use_symmetric_memory(
            get_tp_group(), disabled=not is_allocation_symmetric()
        ):
            layer_input = None
            if (
                envs.SGLANG_DSV4_GFX90A_FUSED_MHC_WEIGHTED_RMS.get()
                and global_batch_size == 1
                and norm_weight is not None
            ):
                assert norm_eps is not None
                norm_weight_bf = (
                    norm_weight.bfloat16()
                    if norm_weight.dtype != torch.bfloat16
                    else norm_weight
                ).contiguous()
                layer_input = gfx90a_mhc_weighted_rmsnorm_triton(
                    residual_cur,
                    pre_cur.squeeze(1),
                    norm_weight_bf,
                    norm_eps,
                )
            if layer_input is None:
                layer_input = mhc_weighted_sum_triton(
                    residual_cur, pre_cur.squeeze(1)
                )
                if layer_input is None:
                    raise RuntimeError(
                        "gfx90a fused MHC weighted-sum rejected its shape"
                    )
                if norm_weight is not None:
                    assert norm_eps is not None
                    norm_weight_bf = (
                        norm_weight.bfloat16()
                        if norm_weight.dtype != torch.bfloat16
                        else norm_weight
                    ).contiguous()
                    normalized = torch.empty_like(layer_input)
                    _gfx90a_mhc_rmsnorm_kernel[(num_tokens,)](
                        layer_input,
                        norm_weight_bf,
                        normalized,
                        hidden_size=hidden_size,
                        eps=norm_eps,
                        BLOCK_H=4096,
                        num_warps=8,
                    )
                    layer_input = normalized
        return (
            residual_cur.view(*outer_shape, hc_mult, hidden_size),
            # Decoder-layer chaining accepts either shape, while the trailing
            # model-level hc_post contract requires (..., hc_mult).
            post_cur.squeeze(1).view(*outer_shape, hc_mult),
            comb_cur.squeeze(1).view(*outer_shape, hc_mult, hc_mult),
            layer_input.view(*outer_shape, hidden_size),
        )

    # The scalar-FMA kernel wins only for small batches where launch
    # overhead dominates; beyond the threshold DeepGEMM's tensor-core path wins.
    fma_token_threshold = 32
    if num_tokens <= fma_token_threshold:
        tile_n = 2 if num_tokens < 8 else 3
        n_splits = 8 if (num_tokens < 8 and hidden_size <= 4096) else 4
    else:
        n_splits = _compute_num_split_for_mhc_pre(num_tokens, hc_hidden_size)

    gemm_out_mul = torch.empty(
        n_splits,
        num_tokens,
        hc_mult3,
        dtype=torch.float32,
        device=residual.device,
    )
    gemm_out_sqrsum = torch.empty(
        n_splits,
        num_tokens,
        dtype=torch.float32,
        device=residual.device,
    )
    residual_cur = torch.empty_like(residual_flat)

    if num_tokens <= fma_token_threshold:
        # Small-batch path: one TileLang launch computes hc_post, the bf16
        # residual write, GEMM partials, and the RMS square-sum partials.
        mhc_fused_post_pre_fma_tilelang(
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            fn.view(hc_mult3, hc_mult, hidden_size),
            gemm_out_mul,
            gemm_out_sqrsum,
            residual_cur,
            hc_mult,
            hidden_size,
            hc_mult3,
            tile_mix_outputs=tile_n,
            split_k=n_splits,
        )
    else:
        # Large-batch path: keep the existing high-throughput TileLang hc_post +
        # DeepGEMM pre-norm GEMM decomposition instead of replacing tensor cores.
        mhc_post_tilelang(
            comb_res_mix.view(num_tokens, hc_mult, hc_mult),
            residual_flat,
            post_layer_mix.view(num_tokens, hc_mult),
            x_flat,
            residual_cur,
            hc_mult,
            hidden_size,
        )

        if envs.SGLANG_OPT_DEEPGEMM_HC_PRENORM.get():
            import deep_gemm

            deep_gemm.tf32_hc_prenorm_gemm(
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul,
                gemm_out_sqrsum,
                num_splits=n_splits,
            )
        else:
            # Fallback mirrors mhc_pre when DeepGEMM prenorm is disabled.
            n_splits = 1
            gemm_out_mul_2d = torch.empty(
                num_tokens, hc_mult3, dtype=torch.float32, device=residual.device
            )
            gemm_out_sqrsum_1d = torch.empty(
                num_tokens, dtype=torch.float32, device=residual.device
            )
            mhc_pre_gemm_sqrsum_tilelang(
                residual_cur.view(num_tokens, hc_hidden_size),
                fn,
                gemm_out_mul_2d,
                gemm_out_sqrsum_1d,
                hc_mult3,
                hc_hidden_size,
            )
            gemm_out_mul = gemm_out_mul_2d.unsqueeze(0)
            gemm_out_sqrsum = gemm_out_sqrsum_1d.unsqueeze(0)

    post_mix_cur = torch.empty(
        num_tokens,
        hc_mult,
        dtype=torch.float32,
        device=residual.device,
    )
    comb_mix_cur = torch.empty(
        num_tokens,
        hc_mult2,
        dtype=torch.float32,
        device=residual.device,
    )
    # layer_input_cur is the post-norm activation fed into the MoE; allocate it
    # in the symmetric memory pool so the Triton inplace MoE runner yields a
    # symmetric all-reduce input (see _mhc_pre_impl).
    with use_symmetric_memory(get_tp_group(), disabled=not is_allocation_symmetric()):
        layer_input_cur = torch.empty(
            num_tokens,
            hidden_size,
            dtype=torch.bfloat16,
            device=residual.device,
        )

    if norm_weight is not None:
        # Final mhc_pre stage: convert GEMM partials into post/comb/layer_input
        # and fuse the following RMSNorm when the model passed a norm weight.
        assert norm_eps is not None
        assert norm_weight.shape == (hidden_size,)
        norm_weight_bf = (
            norm_weight.bfloat16()
            if norm_weight.dtype != torch.bfloat16
            else norm_weight
        )
        if not norm_weight_bf.is_contiguous():
            norm_weight_bf = norm_weight_bf.contiguous()
        mhc_pre_big_fuse_with_norm_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            norm_weight_bf,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            norm_eps,
            n_splits,
            hc_mult,
            hc_mult3,
        )
    else:
        # Same mhc_pre finalization without the model-layer RMSNorm.
        mhc_pre_big_fuse_tilelang(
            gemm_out_mul,
            gemm_out_sqrsum,
            hc_scale,
            hc_base,
            residual_cur,
            post_mix_cur,
            comb_mix_cur,
            layer_input_cur,
            hidden_size,
            rms_eps,
            hc_pre_eps,
            hc_sinkhorn_eps,
            hc_post_mult_value,
            sinkhorn_repeat,
            n_splits,
            hc_mult,
            hc_mult3,
        )

    return (
        residual_cur.view(*outer_shape, hc_mult, hidden_size),
        post_mix_cur.view(*outer_shape, hc_mult, 1),
        comb_mix_cur.view(*outer_shape, hc_mult, hc_mult),
        layer_input_cur.view(*outer_shape, hidden_size),
    )


def npu_hc_pre(
    x: torch.Tensor,
    hc_fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    hc_mult: int,
    hc_sinkhorn_iters: int,
    rms_norm_eps: float,
    hc_eps: float,
    forward_batch=None,
) -> tuple:
    """NPU-accelerated hc_pre via the custom_ops kernel.

    Returns (y, post, comb, norm_fused).  norm_fused is always False
    because npu_hc_pre does not fold input_layernorm — the caller must
    apply it separately.
    """
    shape, dtype = x.size(), x.dtype

    # IDLE / empty short-circuit, mirroring the dsv4-flash source.
    # The kernel emits post/comb in fp32 (sinkhorn iterates in fp32),
    # so the dummies must too — otherwise downstream comb/post-aware
    # ops see a silent fp32 ↔ bf16 split between idle and non-idle
    # batches.
    is_idle = forward_batch is not None and forward_batch.forward_mode.is_idle()
    if is_idle or x.shape[0] == 0:
        bs = x.shape[0]
        y = torch.empty((bs, shape[-1]), dtype=dtype, device=x.device)
        post = torch.empty((bs, hc_mult), dtype=torch.float32, device=x.device)
        comb = torch.empty(
            (bs, hc_mult, hc_mult),
            dtype=torch.float32,
            device=x.device,
        )
        return y, post, comb, False

    # Note the return order: (y, post, comb) — y is the (T, hidden)
    # mixed activation, post / comb are the hc_post inputs. The
    # fused kernel emits y in fp32 (sinkhorn iterates in fp32), so
    # cast back to the input dtype before the downstream
    # aclnnRmsNorm (which has no x=fp32 / gamma=bf16 overload).
    y, post, comb = torch.ops.custom.npu_hc_pre(
        x,
        hc_fn,
        hc_scale,
        hc_base,
        hc_mult=hc_mult,
        hc_sinkhorn_iters=hc_sinkhorn_iters,
        norm_eps=rms_norm_eps,
        hc_eps=hc_eps,
    )
    # npu_hc_pre uses norm_eps for sinkhorn's internal RMS only; it does
    # not fold input_layernorm. Return norm_fused=False so the caller
    # applies the layernorm itself, matching the deepgemm/torch paths.
    return y.to(dtype), post, comb, False
