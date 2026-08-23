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
                "run_static",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_static",
            ),
            (
                "run_static_nomask",
                f"sglang::Gfx90aFp4ExpertGateUpKernel<{args}>::run_static_nomask",
            ),
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
                "run_static",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_static",
            ),
            (
                "run_static_nomask",
                f"sglang::Gfx90aFp4ExpertDownKernel<{args}>::run_static_nomask",
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
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = x.shape
    i, t = two_i // 2, expert_ids.shape[1]
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    out = torch.empty((m, t, i), dtype=torch.bfloat16, device=x.device)
    kernel = _jit_gate_up(e, m, t, ge, i, k)
    args = [
        x,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, two_i, k // 32),
        expert_ids,
    ]
    if expert_mask is not None:
        args.append(expert_mask)
    if live_count is None:
        if expert_mask is None:
            kernel.run_static_nomask(*args, out, float(limit))
        else:
            kernel.run_static(*args, out, float(limit))
    else:
        assert expert_mask is not None, "dynamic live count requires an expert mask"
        kernel.run(*args, live_count, out, float(limit))
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
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, t, k = x.shape
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    kernel = _jit_down(e, m, t, ge, n, k)
    args = [
        x,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, n, k // 32),
        expert_ids,
        topk_weights,
    ]
    if expert_mask is not None:
        args.insert(4, expert_mask)
    if live_count is None:
        if expert_mask is None:
            kernel.run_static_nomask(*args, out)
        else:
            kernel.run_static(*args, out)
    else:
        assert expert_mask is not None, "dynamic live count requires an expert mask"
        kernel.run(*args, live_count, out)
    return out
