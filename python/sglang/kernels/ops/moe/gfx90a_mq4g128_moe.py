from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _indexed_module(e: int, m: int, t: int, n: int, k: int) -> Module:
    args = make_cpp_args(e, m, t, n, k)
    return load_jit(
        "gfx90a_mq4g128_indexed",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aMq4g128Indexed<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _grouped_module(
    e: int, m: int, t: int, n: int, k: int, assignments: int, groups: int
) -> Module:
    args = make_cpp_args(e, m, t, n, k, assignments, groups)
    return load_jit(
        "gfx90a_mq4g128_grouped",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aMq4g128Grouped<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _sorter_module(e: int, m: int, t: int, assignments: int) -> Module:
    args = make_cpp_args(e, m, t, assignments)
    return load_jit(
        "gfx90a_mq4g128_sorter",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aMq4g128Sorter<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def mq4g128_indexed(
    x: torch.Tensor, weight: torch.Tensor, expert_ids: torch.Tensor
) -> torch.Tensor:
    m, k = x.shape
    e, n, groups, block_bytes = weight.shape
    assert x.dtype == torch.float32 and x.is_contiguous()
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert groups * 128 == k and block_bytes == 72
    assert expert_ids.shape[0] == m and expert_ids.dtype == torch.int32
    t = expert_ids.shape[1]
    out = torch.empty((m, t, n), dtype=torch.float32, device=x.device)
    _indexed_module(e, m, t, n, k).run(x, weight, expert_ids, out)
    return out


def build_expert_a4_runs(
    expert_ids: torch.Tensor, num_experts: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Graph-safe static-buffer HIP histogram/scan sorter."""
    m, t = expert_ids.shape
    capacity = m * t
    sorted_assignments = torch.empty(
        capacity * 4, dtype=torch.int32, device=expert_ids.device
    )
    sorted_experts = torch.empty(
        capacity, dtype=torch.int32, device=expert_ids.device
    )
    _sorter_module(num_experts, m, t, 4).run(
        expert_ids, sorted_assignments, sorted_experts
    )
    return sorted_assignments, sorted_experts


def mq4g128_grouped(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    m, k = x.shape
    e, n, groups, block_bytes = weight.shape
    assert groups * 128 == k and block_bytes == 72
    t = expert_ids.shape[1]
    sorted_ids, sorted_experts = build_expert_a4_runs(expert_ids, e)
    out = torch.empty((m, t, n), dtype=torch.float32, device=x.device)
    _grouped_module(e, m, t, n, k, 4, sorted_experts.numel()).run(
        x, weight, sorted_ids, sorted_experts, out
    )
    return out
