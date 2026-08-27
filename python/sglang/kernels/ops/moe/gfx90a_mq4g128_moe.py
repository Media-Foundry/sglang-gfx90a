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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Correctness-first expert run builder; graph-safe sorter follows later."""
    flat = expert_ids.reshape(-1)
    valid = (flat >= 0) & (flat < num_experts)
    assignments = torch.arange(flat.numel(), dtype=torch.int32, device=flat.device)[valid]
    experts = flat[valid]
    order = torch.argsort(experts, stable=True)
    assignments = assignments[order]
    experts = experts[order]
    counts = torch.bincount(experts.to(torch.int64), minlength=num_experts)
    run_experts = torch.repeat_interleave(
        torch.arange(num_experts, dtype=torch.int32, device=flat.device),
        torch.div(counts + 3, 4, rounding_mode="floor"),
    )
    group_count = run_experts.numel()
    # Per-expert A4 padding means the worst case is one group per assignment
    # (all assignments select distinct experts), not ceil(total / 4).
    capacity = flat.numel()
    padded = torch.full((capacity * 4,), -1, dtype=torch.int32, device=flat.device)
    cursor = 0
    group = 0
    for expert in range(num_experts):
        count = int(counts[expert].item())
        for start in range(0, count, 4):
            take = min(4, count - start)
            padded[group * 4 : group * 4 + take] = assignments[cursor + start : cursor + start + take]
            group += 1
        cursor += count
    padded_experts = torch.full((capacity,), -1, dtype=torch.int32, device=flat.device)
    padded_experts[:group_count] = run_experts
    return padded, padded_experts, torch.tensor([group_count], dtype=torch.int32, device=flat.device)


def mq4g128_grouped(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    m, k = x.shape
    e, n, groups, block_bytes = weight.shape
    assert groups * 128 == k and block_bytes == 72
    t = expert_ids.shape[1]
    sorted_ids, sorted_experts, _ = build_expert_a4_runs(expert_ids, e)
    out = torch.empty((m, t, n), dtype=torch.float32, device=x.device)
    _grouped_module(e, m, t, n, k, 4, sorted_experts.numel()).run(
        x, weight, sorted_ids, sorted_experts, out
    )
    return out
