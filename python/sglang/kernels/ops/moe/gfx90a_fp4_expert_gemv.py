from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gate_up(e: int, m: int, t: int, ge: int, i: int, k: int) -> Module:
    args = make_cpp_args(e, m, t, ge, i, k, 2, 8)
    return load_jit(
        "gfx90a_fp4_expert_gate_up",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run"),
            (
                "run_prequant",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_prequant",
            ),
            (
                "run_static",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_static",
            ),
            (
                "run_prequant_static",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_prequant_static",
            ),
            (
                "run_static_nomask",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_static_nomask",
            ),
            (
                "run_prequant_static_nomask",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_prequant_static_nomask",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gate_up_grouped(
    e: int, m: int, t: int, i: int, k: int, assignments: int
) -> Module:
    args = make_cpp_args(e, m, t, i, k, assignments, 2, 8)
    return load_jit(
        "gfx90a_fp4_expert_gate_up_grouped",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4ExpertGateUpGroupedKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down(e: int, m: int, t: int, ge: int, n: int, k: int) -> Module:
    args = make_cpp_args(e, m, t, ge, n, k, 2, 8)
    return load_jit(
        "gfx90a_fp4_expert_down",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run"),
            (
                "run_prequant",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_prequant",
            ),
            (
                "run_static",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_static",
            ),
            (
                "run_prequant_static",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_prequant_static",
            ),
            (
                "run_static_nomask",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_static_nomask",
            ),
            (
                "run_prequant_static_nomask",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_prequant_static_nomask",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_fp4_expert_gate_up(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_mask: torch.Tensor | None,
    live_count: torch.Tensor | None,
    limit: float,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = x.shape
    i, t = two_i // 2, expert_ids.shape[1]
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    out = torch.empty((m, t, i), dtype=torch.bfloat16, device=x.device)
    kernel = _jit_gate_up(e, m, t, ge, i, k)
    args = [x]
    if prequant is not None:
        xq, x_scale = prequant
        assert xq.shape == x.shape and xq.dtype == torch.int8
        assert x_scale.shape == (m, k // 32) and x_scale.dtype == torch.float32
        args.extend((xq, x_scale))
    args.extend(
        (
            weight.view(torch.uint8),
            weight_scale.view(torch.uint8).reshape(e, two_i, k // 32),
            expert_ids,
        )
    )
    if expert_mask is not None:
        args.append(expert_mask)
    if live_count is None:
        if expert_mask is None:
            getattr(
                kernel,
                (
                    "run_prequant_static_nomask"
                    if prequant is not None
                    else "run_static_nomask"
                ),
            )(*args, out, float(limit))
        else:
            getattr(
                kernel,
                "run_prequant_static" if prequant is not None else "run_static",
            )(*args, out, float(limit))
    else:
        assert expert_mask is not None, "dynamic live count requires an expert mask"
        getattr(kernel, "run_prequant" if prequant is not None else "run")(
            *args, live_count, out, float(limit)
        )
    return out


def gfx90a_fp4_expert_gate_up_grouped(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    limit: float,
    assignments: int = 4,
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = xq.shape
    i = two_i // 2
    assert packed_k * 2 == k
    assert xq.dtype == torch.int8 and xq.is_contiguous()
    assert x_scale.shape == (m, k // 32) and x_scale.dtype == torch.float32
    assert sorted_ids.dtype == torch.int32 and sorted_ids.is_contiguous()
    assert sorted_expert_ids.dtype == torch.int32
    assert num_valid_ids.shape == (2,) and num_valid_ids.dtype == torch.int32
    out = torch.empty((m, topk, i), dtype=torch.bfloat16, device=xq.device)
    _jit_gate_up_grouped(e, m, topk, i, k, assignments).run(
        xq,
        x_scale,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, two_i, k // 32),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        float(limit),
    )
    return out


def gfx90a_fp4_expert_down(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_mask: torch.Tensor | None,
    topk_weights: torch.Tensor,
    live_count: torch.Tensor | None,
    out: torch.Tensor | None = None,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, t, k = x.shape
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    kernel = _jit_down(e, m, t, ge, n, k)
    args = [x]
    if prequant is not None:
        xq, x_scale = prequant
        assert xq.shape == x.shape and xq.dtype == torch.int8
        assert x_scale.shape == (m, t, k // 32) and x_scale.dtype == torch.float32
        args.extend((xq, x_scale))
    args.extend(
        (
            weight.view(torch.uint8),
            weight_scale.view(torch.uint8).reshape(e, n, k // 32),
            expert_ids,
        )
    )
    if expert_mask is not None:
        args.append(expert_mask)
    args.append(topk_weights)
    if live_count is None:
        if expert_mask is None:
            getattr(
                kernel,
                (
                    "run_prequant_static_nomask"
                    if prequant is not None
                    else "run_static_nomask"
                ),
            )(*args, out)
        else:
            getattr(
                kernel,
                "run_prequant_static" if prequant is not None else "run_static",
            )(*args, out)
    else:
        assert expert_mask is not None, "dynamic live count requires an expert mask"
        getattr(kernel, "run_prequant" if prequant is not None else "run")(
            *args, live_count, out
        )
    return out
