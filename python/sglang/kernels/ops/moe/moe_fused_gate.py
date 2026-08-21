from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Tuple

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import cache_once, is_arch_support_pdl, load_jit
from sglang.kernels.kernel_api_logging import debug_kernel_api
from sglang.kernels.ops.moe import moe_route_radix

if TYPE_CHECKING:
    from tvm_ffi.module import Module

_SCORING_FUNC_MAP = {
    "sigmoid": 0,
    "sqrtsoftplus": 1,
    "softmax": 2,
}
_gfx90a_router_diag_logged = False


@triton.jit
def _gfx90a_fp32_ordered_key(value):
    bits = value.to(tl.uint32, bitcast=True)
    sign = tl.full(bits.shape, 0x80000000, tl.uint32)
    full = tl.full(bits.shape, 0xFFFFFFFF, tl.uint32)
    return bits ^ tl.where((bits & sign) != 0, full, sign)


@triton.jit
def _gfx90a_sqrtsoftplus_topk_kernel(
    scores,
    bias,
    weights,
    indices,
    routed_scaling_factor: tl.constexpr,
    APPLY_SCALE: tl.constexpr,
):
    offs = tl.arange(0, 256)
    logits = tl.load(scores + offs).to(tl.float32)
    bias_v = tl.load(bias + offs).to(tl.float32)
    softplus = tl.where(logits > 20.0, logits, tl.log(1.0 + tl.exp(logits)))
    activated = tl.sqrt(softplus)
    ranked = activated + bias_v
    ranked = tl.where(ranked == ranked, ranked, -float("inf"))

    # Pack the monotonic FP32 key and inverse expert id. Larger packed values
    # win, so equal scores deterministically select the lower expert id.
    value_key = _gfx90a_fp32_ordered_key(ranked).to(tl.uint64)
    packed = (value_key << 16) | (256 - offs).to(tl.uint64)
    winners = tl.topk(packed, 8)
    winner_ids = (256 - (winners & 0xFFFF).to(tl.int32)).to(tl.int32)

    k = tl.arange(0, 8)
    selected_weight = tl.sum(
        tl.where(offs[None, :] == winner_ids[:, None], activated[None, :], 0.0),
        axis=1,
    )
    routed_sum = tl.sum(tl.where(k < 6, selected_weight, 0.0), axis=0)
    scale = routed_scaling_factor if APPLY_SCALE else 1.0
    mask = k < 6
    tl.store(weights + k, selected_weight / routed_sum * scale, mask=mask)
    tl.store(indices + k, winner_ids, mask=mask)


def gfx90a_sqrtsoftplus_topk_triton(
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
    _gfx90a_sqrtsoftplus_topk_kernel[(1,)](
        scores,
        bias,
        weights,
        indices,
        routed_scaling_factor=float(routed_scaling_factor),
        APPLY_SCALE=bool(apply_scale),
        num_warps=1,
    )
    return weights, indices


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[("moe_fused_gate", "MoEFusedGateKernel::run")],
    )


@cache_once
def can_use_moe_fused_gate() -> bool:
    logger = logging.getLogger(__name__)
    try:
        _jit_moe_fused_gate_module()
        return True
    except Exception as e:
        logger.warning(f"Failed to load JIT MoE fused gate kernel: {e}")
        return False


def moe_fused_gate_jit(
    input: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"

    assert input.dtype == torch.float32, "input must be float32"
    assert bias.dtype == torch.float32, "bias must be float32"
    assert input.ndim == 2, "input must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert input.size(1) == bias.size(0), "input and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"

    num_rows, _ = input.shape
    device = input.device

    output = torch.empty(num_rows, topk, dtype=torch.float32, device=device)
    indices = torch.empty(num_rows, topk, dtype=torch.int32, device=device)

    module = _jit_moe_fused_gate_module()
    module.moe_fused_gate(
        input,
        bias,
        output,
        indices,
        topk,
        scoring_func_int,
        num_fused_shared_experts,
        renormalize,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return output, indices


@triton.jit
def _router_triton_kernel(
    scores_ptr,  # [M, N] fp32, GEMM output (raw logits)
    bias_ptr,  # [N]    fp32/fp16/bf16 (upcast to fp32 on load)
    out_weights_ptr,  # [M, K] fp32
    out_indices_ptr,  # [M, K] int32
    M,
    routed_scaling_factor,
    moe_softcapping,
    N: tl.constexpr,
    K: tl.constexpr,  # total topk (includes fused shared experts)
    K_ROUTED: tl.constexpr,  # K - num_fused_shared_experts
    BLOCK_M: tl.constexpr,  # rows processed per program (row tiling)
    BLOCK_N: tl.constexpr,  # >= N, power of 2
    BLOCK_K: tl.constexpr,  # >= K, power of 2
    N_GROUP: tl.constexpr,  # expert groups (1 = ungrouped)
    TOPK_GROUP: tl.constexpr,  # groups kept per token (grouped routing)
    EXPERTS_PER_GROUP: tl.constexpr,  # N // N_GROUP
    BLOCK_G: tl.constexpr,  # >= N_GROUP, power of 2
    SCORING_FUNC: tl.constexpr,  # 0 = sigmoid, 1 = sqrtsoftplus, 2 = softmax
    HAS_SOFTCAP: tl.constexpr,  # tanh softcapping (softmax only)
    RENORMALIZE: tl.constexpr,
    APPLY_SCALE: tl.constexpr,  # apply_routed_scaling_factor_on_output
    USE_PDL: tl.constexpr,
    stride_sm,
    stride_sn,
    stride_wm,
    stride_wk,
    stride_im,
    stride_ik,
) -> None:
    # Row-tiled: each program handles BLOCK_M rows; all reductions run along the
    # expert (N) axis. Tiling rows keeps CTAs large enough to stay occupancy-bound
    # rather than launch-bound at small N (many tiny 1-warp CTAs otherwise).
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)  # [BLOCK_M]
    offs_n = tl.arange(0, BLOCK_N)  # [BLOCK_N]
    mask_m = offs_m < M
    mask_n = offs_n < N

    # prefetch bias before PDL wait
    bias = tl.load(bias_ptr + offs_n, mask=mask_n, other=0.0).to(
        tl.float32
    )  # [BLOCK_N]

    if USE_PDL:
        tl.extra.cuda.gdc_wait()

    row_ptr = scores_ptr + offs_m[:, None] * stride_sm + offs_n[None, :] * stride_sn
    mask2d = mask_m[:, None] & mask_n[None, :]
    scores = tl.load(row_ptr, mask=mask2d, other=0.0).to(
        tl.float32
    )  # [BLOCK_M, BLOCK_N]

    if SCORING_FUNC == 0:
        # sigmoid(x) = 1 / (1 + exp(-x)); bias is for ranking only, weight is bias-free.
        activated = tl.sigmoid(scores)
        biased = activated + bias[None, :]
    elif SCORING_FUNC == 1:
        # sqrt(softplus(x)) = sqrt(log1p(exp(x))); guard against overflow when x is large.
        sp = tl.where(scores > 20.0, scores, tl.log(1.0 + tl.exp(scores)))
        activated = tl.sqrt(sp)
        biased = activated + bias[None, :]
    else:
        # softmax over the row: weight is the softmax probability (bias kept), with
        # optional tanh softcapping. Ranking by the (softcapped, biased) logit is
        # monotonic with the softmax prob, so the topk loop below ranks on `biased`.
        logit = scores
        if HAS_SOFTCAP:
            # tanh(z) = 2*sigmoid(2z) - 1 (avoids relying on tl.math.tanh availability).
            z = logit / moe_softcapping
            logit = moe_softcapping * (2.0 * tl.sigmoid(2.0 * z) - 1.0)
        biased = logit + bias[None, :]
        biased = tl.where(mask_n[None, :], biased, -float("inf"))
        row_max = tl.max(biased, axis=1)[:, None]  # [BLOCK_M, 1]
        exp_row = tl.where(mask_n[None, :], tl.exp(biased - row_max), 0.0)
        row_sum = tl.sum(exp_row, axis=1)[:, None]  # [BLOCK_M, 1]
        activated = exp_row / row_sum

    biased = tl.where(mask_n[None, :], biased, -float("inf"))  # [BLOCK_M, BLOCK_N]

    # Map NaN -> a finite floor
    biased = tl.where(biased == biased, biased, -1e30)  # [BLOCK_M, BLOCK_N]

    # Grouped routing (DeepSeek-V3 noaux_tc): per-group score = sum of the top-2
    # biased values; keep TOPK_GROUP groups (lowest group id wins ties); mask the
    # experts of dropped groups to -inf before the top-k below. Weight is still the
    # bias-free `activated`. Constexpr N_GROUP <= 1 skips this entirely (ungrouped).
    if N_GROUP > 1:
        offs_g = tl.arange(0, BLOCK_G)  # [BLOCK_G]
        group_of_n = offs_n // EXPERTS_PER_GROUP  # [BLOCK_N]
        group_score = tl.full([BLOCK_M, BLOCK_G], -float("inf"), dtype=tl.float32)
        for g in tl.static_range(N_GROUP):
            in_g = (group_of_n[None, :] == g) & mask_n[None, :]
            vals = tl.where(in_g, biased, -float("inf"))
            top1 = tl.max(vals, axis=1)[:, None]  # [BLOCK_M, 1]
            vals2 = tl.where(vals >= top1, -float("inf"), vals)
            top2 = tl.max(vals2, axis=1)[:, None]  # [BLOCK_M, 1]
            group_score = tl.where(offs_g[None, :] == g, top1 + top2, group_score)

        gcur = group_score
        keep = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        for _i in tl.static_range(TOPK_GROUP):
            gmax = tl.max(gcur, axis=1)[:, None]  # [BLOCK_M, 1]
            glane = tl.where(gcur == gmax, offs_g[None, :], N_GROUP + 1)
            win_g = tl.min(glane, axis=1)[:, None]  # [BLOCK_M, 1] lowest-id on ties
            keep = tl.where(group_of_n[None, :] == win_g, 1.0, keep)
            gcur = tl.where(offs_g[None, :] == win_g, -float("inf"), gcur)
        biased = tl.where(keep > 0.0, biased, -float("inf"))

    offs_k = tl.arange(0, BLOCK_K)  # [BLOCK_K]
    mask_k_total = offs_k < K
    mask_k_routed = offs_k < K_ROUTED
    selected_vals = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.float32)
    selected_idx = tl.zeros([BLOCK_M, BLOCK_K], dtype=tl.int32)

    cur = biased  # [BLOCK_M, BLOCK_N]
    for k in tl.static_range(K_ROUTED):
        max_val = tl.max(cur, axis=1)[:, None]  # [BLOCK_M, 1]
        is_max = cur == max_val
        lane_id = tl.where(is_max, offs_n[None, :], N + 1)  # lowest expert id wins ties
        win_lane = tl.min(lane_id, axis=1)[:, None].to(tl.int32)  # [BLOCK_M, 1]
        win_activated = tl.sum(
            tl.where(offs_n[None, :] == win_lane, activated, 0.0), axis=1
        )[
            :, None
        ]  # [BLOCK_M, 1]
        slot = offs_k[None, :] == k  # [1, BLOCK_K]
        selected_vals = tl.where(slot, win_activated, selected_vals)
        selected_idx = tl.where(slot, win_lane, selected_idx)
        cur = tl.where(offs_n[None, :] == win_lane, -float("inf"), cur)

    routed_sum = tl.sum(tl.where(mask_k_routed[None, :], selected_vals, 0.0), axis=1)[
        :, None
    ]  # [BLOCK_M, 1]

    # Fill fused-shared-expert slots: weight = routed_sum / routed_scaling_factor,
    # id = num_experts + (slot - K_ROUTED).
    if K_ROUTED < K:
        is_shared = (offs_k[None, :] >= K_ROUTED) & mask_k_total[None, :]
        shared_weight = routed_sum / routed_scaling_factor  # [BLOCK_M, 1]
        shared_idx = (N + (offs_k - K_ROUTED)).to(tl.int32)[None, :]  # [1, BLOCK_K]
        selected_vals = tl.where(is_shared, shared_weight, selected_vals)
        selected_idx = tl.where(is_shared, shared_idx, selected_idx)

    if USE_PDL:
        tl.extra.cuda.gdc_launch_dependents()

    if RENORMALIZE:
        norm = tl.where(routed_sum > 0.0, routed_sum, 1.0)  # [BLOCK_M, 1]
        selected_vals = selected_vals / norm
    if APPLY_SCALE:
        selected_vals = selected_vals * routed_scaling_factor

    out_w_ptr = (
        out_weights_ptr + offs_m[:, None] * stride_wm + offs_k[None, :] * stride_wk
    )
    out_i_ptr = (
        out_indices_ptr + offs_m[:, None] * stride_im + offs_k[None, :] * stride_ik
    )
    store_mask = mask_m[:, None] & mask_k_total[None, :]
    tl.store(out_w_ptr, selected_vals, mask=store_mask)
    tl.store(out_i_ptr, selected_idx, mask=store_mask)


@debug_kernel_api
def moe_fused_gate(
    scores: torch.Tensor,
    bias: torch.Tensor,
    topk: int,
    scoring_func: str = "sigmoid",
    num_fused_shared_experts: int = 0,
    renormalize: bool = True,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
    moe_softcapping: float = 0.0,
    num_expert_group: int = 1,
    topk_group: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton fused router: scoring + bias + topk + (optional) renorm/scale.

    Mirrors the semantics of :func:`moe_fused_gate_jit` (the CUDA JIT kernel).
    With ``num_expert_group > 1`` it performs DeepSeek-V3 grouped routing
    (per-group top-2-sum group scores, keep ``topk_group`` groups, then top-k
    within). The first argument is named ``scores`` (raw GEMM logits) to match
    the existing call sites.
    """
    scoring_func_int = _SCORING_FUNC_MAP.get(scoring_func.lower())
    assert (
        scoring_func_int is not None
    ), f"Unknown scoring_func '{scoring_func}', must be one of {list(_SCORING_FUNC_MAP.keys())}"
    assert scores.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "scores must be float32/float16/bfloat16"
    # The kernel loads the bias and upcasts it to fp32 in-register (see
    # _router_triton_kernel), so a non-fp32 bias (DeepSeek-V4 stores the
    # correction bias in bf16) needs no host-side cast/copy.
    assert bias.dtype in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ), "bias must be float32/float16/bfloat16"
    assert scores.ndim == 2, "scores must be 2D"
    assert bias.ndim == 1, "bias must be 1D"
    assert scores.size(1) == bias.size(0), "scores and bias must have same num_experts"
    assert topk > num_fused_shared_experts, "topk must be > num_fused_shared_experts"
    if routed_scaling_factor is None:
        routed_scaling_factor = 1.0

    global _gfx90a_router_diag_logged
    from sglang.srt.environ import envs

    if (
        envs.SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER.get()
        and topk == 6
        and not _gfx90a_router_diag_logged
    ):
        logging.getLogger(__name__).warning(
            "gfx90a router probe: shape=%s scores=%s bias=%s scoring=%s "
            "fused_shared=%s renorm=%s groups=%s topk_group=%s softcap=%s scale_out=%s",
            tuple(scores.shape),
            scores.dtype,
            bias.dtype,
            scoring_func,
            num_fused_shared_experts,
            renormalize,
            num_expert_group,
            topk_group,
            moe_softcapping,
            apply_routed_scaling_factor_on_output,
        )
        _gfx90a_router_diag_logged = True

    # CDNA2 decode specialization.  Keep the GEMM producing ``scores`` on its
    # multi-CU path; this kernel replaces only grouped selection/renormalize.
    if (
        scores.shape == (1, 256)
        and bias.shape == (256,)
        and bias.dtype in (torch.bfloat16, torch.float32)
        and topk == 6
        and scoring_func_int == 0
        and num_fused_shared_experts == 0
        and renormalize
        and num_expert_group == 8
        and topk_group == 4
        and moe_softcapping == 0.0
    ):
        if envs.SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER.get():
            from sglang.kernels.ops.moe.gfx90a_grouped_router import (
                gfx90a_grouped_router,
            )

            native = gfx90a_grouped_router(
                scores,
                bias,
                float(routed_scaling_factor),
                bool(apply_routed_scaling_factor_on_output),
            )
            if native is not None:
                return native

    if (
        scores.shape == (1, 256)
        and bias.shape == (256,)
        and bias.dtype in (torch.bfloat16, torch.float32)
        and topk == 6
        and scoring_func_int == 1
        and num_fused_shared_experts == 0
        and renormalize
        and num_expert_group <= 1
        and moe_softcapping == 0.0
    ):
        if envs.SGLANG_DSV4_GFX90A_TRITON_TOPK_ROUTER.get():
            triton_topk = gfx90a_sqrtsoftplus_topk_triton(
                scores,
                bias,
                float(routed_scaling_factor),
                bool(apply_routed_scaling_factor_on_output),
            )
            if triton_topk is not None:
                return triton_topk
        if envs.SGLANG_DSV4_GFX90A_NATIVE_GROUPED_ROUTER.get():
            from sglang.kernels.ops.moe.gfx90a_grouped_router import (
                gfx90a_sqrtsoftplus_router,
                preload_gfx90a_router,
            )
            from sglang.srt.model_executor.runner import get_is_capture_mode

            graph_warmup = (
                get_is_capture_mode()
                and not torch.cuda.is_current_stream_capturing()
            )
            preload_gfx90a_router()
            if not graph_warmup:
                native = gfx90a_sqrtsoftplus_router(
                    scores,
                    bias,
                    float(routed_scaling_factor),
                    bool(apply_routed_scaling_factor_on_output),
                )
                if native is not None:
                    return native

    # K3 radix-select fast path: native-CUDA radix-select replaces the 16
    # dependent argmax rounds (single CTA per token; ids bit-identical to this
    # triton kernel incl. ties).
    # The radix kernel keeps keys register-resident and returns winners in
    # expert-id order (skipping the biased-descending sort; downstream MoE
    # kernels are order-insensitive). It is 3.1-3.5x faster than the Triton
    # kernel at [1..8192, 896] top-16 on B200.
    if (
        scoring_func.lower() == "sigmoid"
        and num_fused_shared_experts == 0
        and num_expert_group <= 1
        and moe_softcapping == 0.0
    ):
        radix_args = (
            scores,
            bias,
            topk,
            renormalize,
            routed_scaling_factor,
            apply_routed_scaling_factor_on_output,
        )
        if moe_route_radix.covered(scores, bias, topk):
            return moe_route_radix.route_radix(*radix_args, sorted=False)

    M, N = scores.shape
    K = topk
    K_routed = topk - num_fused_shared_experts
    if num_expert_group > 1:
        assert N % num_expert_group == 0, "num_experts must be divisible by group count"
        assert 1 <= topk_group <= num_expert_group, "invalid topk_group"
    experts_per_group = N // num_expert_group
    BLOCK_G = triton.next_power_of_2(num_expert_group)

    weights = torch.empty((M, K), dtype=torch.float32, device=scores.device)
    indices = torch.empty((M, K), dtype=torch.int32, device=scores.device)

    BLOCK_N = triton.next_power_of_2(N)  # 256 -> 256, 384 -> 512
    BLOCK_K = triton.next_power_of_2(K)  # 6 -> 8, 8 -> 8
    # Single warp per program keeps the per-row top-k reductions on cheap warp
    # shuffles; pack a few rows per program only when N is small so tiny launches
    # stay occupancy-bound. Swept on H100/B200: this beats the AOT kernels across
    # shapes, whereas larger tiles / more warps regress (register pressure).
    BLOCK_M = max(1, min(4, 256 // BLOCK_N))
    # For wide rows (e.g. Kimi K3: 896 experts, BLOCK_N 1024) the K sequential
    # argmax passes dominate and benefit from more warps despite the
    # cross-warp reduction cost.
    num_warps = 1 if BLOCK_N <= 512 else 4
    if (
        torch.version.hip
        and M == 1
        and N == 256
        and K == 6
        and scoring_func_int == 1
        and getattr(
            torch.cuda.get_device_properties(scores.device), "gcnArchName", ""
        ).split(":", 1)[0]
        == "gfx90a"
    ):
        num_warps = envs.SGLANG_DSV4_GFX90A_ROUTER_NUM_WARPS.get()
    grid = (triton.cdiv(M, BLOCK_M),)
    use_pdl = is_arch_support_pdl()
    extra = {"launch_pdl": True} if use_pdl else {}
    _router_triton_kernel[grid](
        scores,
        bias,
        weights,
        indices,
        M,
        float(routed_scaling_factor),
        float(moe_softcapping),
        N=N,
        K=K,
        K_ROUTED=K_routed,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        N_GROUP=num_expert_group,
        TOPK_GROUP=topk_group,
        EXPERTS_PER_GROUP=experts_per_group,
        BLOCK_G=BLOCK_G,
        SCORING_FUNC=scoring_func_int,
        HAS_SOFTCAP=bool(moe_softcapping != 0.0),
        RENORMALIZE=bool(renormalize),
        APPLY_SCALE=bool(apply_routed_scaling_factor_on_output),
        USE_PDL=use_pdl,
        stride_sm=scores.stride(0),
        stride_sn=scores.stride(1),
        stride_wm=weights.stride(0),
        stride_wk=weights.stride(1),
        stride_im=indices.stride(0),
        stride_ik=indices.stride(1),
        num_warps=num_warps,
        **extra,
    )
    return weights, indices
