from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _remap_module(m: int, t: int, e: int) -> Module:
    args = make_cpp_args(m, t, e)
    return load_jit(
        "gfx90a_mq4g128_remap_topk",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aMq4g128RemapTopk<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def mq4g128_remap_topk(
    expert_ids: torch.Tensor, local_expert_mapping: torch.Tensor
) -> torch.Tensor:
    m, t = expert_ids.shape
    assert expert_ids.dtype == torch.int32 and expert_ids.is_contiguous()
    assert (
        local_expert_mapping.dtype == torch.int32
        and local_expert_mapping.is_contiguous()
    )
    out = torch.empty_like(expert_ids)
    _remap_module(m, t, local_expert_mapping.numel()).run(
        expert_ids, local_expert_mapping, out
    )
    return out


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
def _expert_owned_sorter_module(e: int, m: int, t: int) -> Module:
    args = make_cpp_args(e, m, t)
    return load_jit(
        "gfx90a_mq4g128_expert_owned_sorter",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aMq4g128ExpertOwnedSorter<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _expert_owned_module(
    e: int,
    m: int,
    t: int,
    n: int,
    k: int,
    waves: int = 2,
    symmetric: bool = False,
) -> Module:
    args = make_cpp_args(e, m, t, n, k, waves, symmetric)
    suffix = "symmetric" if symmetric else "affine"
    return load_jit(
        f"gfx90a_mq4g128_expert_owned_{suffix}",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aMq4g128ExpertOwned<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _persistent_slots_module(e: int, m: int, t: int, n: int, k: int) -> Module:
    args = make_cpp_args(e, m, t, n, k)
    return load_jit(
        "gfx90a_mq4g128_persistent_slots",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aMq4g128PersistentSlots<{args}>::run")
        ],
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


@cache_once
def _weighted_reduce_module(t: int, n: int) -> Module:
    args = make_cpp_args(t, n)
    return load_jit(
        "gfx90a_mq4g128_weighted_reduce",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aMq4g128WeightedReduce<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _masked_weighted_reduce_module(m: int, t: int, n: int) -> Module:
    args = make_cpp_args(m, t, n)
    return load_jit(
        "gfx90a_mq4g128_masked_weighted_reduce",
        *args,
        cuda_files=["moe/gfx90a_mq4g128_moe.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aMq4g128MaskedWeightedReduce<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def mq4g128_indexed(
    x: torch.Tensor,
    weight: torch.Tensor,
    expert_ids: torch.Tensor,
    zero_invalid: bool = True,
) -> torch.Tensor:
    m, k = x.shape
    e, n, groups, block_bytes = weight.shape
    assert x.dtype == torch.float32 and x.is_contiguous()
    assert weight.dtype == torch.uint8 and weight.is_contiguous()
    assert groups * 128 == k and block_bytes == 72
    assert expert_ids.shape[0] == m and expert_ids.dtype == torch.int32
    t = expert_ids.shape[1]
    use_expert_owned_m32 = (
        os.environ.get("SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M32", "0") == "1"
        and e == 128
        and (
            (m, t, n, k) == (32, 10, 1280, 2560)
            or (m, t, n, k) == (320, 1, 2560, 640)
        )
    )
    use_expert_owned_m64 = (
        os.environ.get("SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M64", "0") == "1"
        and e == 128
        and (
            (m, t, n, k) == (64, 10, 1280, 2560)
            or (m, t, n, k) == (640, 1, 2560, 640)
        )
    )
    use_expert_owned_m128 = (
        os.environ.get("SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M128", "0") == "1"
        and e == 128
        and (
            (m, t, n, k) == (128, 10, 1280, 2560)
            or (m, t, n, k) == (1280, 1, 2560, 640)
        )
    )
    use_expert_owned_m16 = (
        os.environ.get(
            "SGLANG_QWEN4_GFX90A_MQ4G128_EXPERT_OWNED_M16", "0"
        )
        == "1"
        # TP4/EP4 owns 128 experts per rank.  The TP2xPP2/EP2 pipeline
        # profile owns 256 experts per rank but presents the same M16 gate
        # and flattened-down shapes to each stage; the sorter/projection are
        # already templated by E and remain graph-static for either layout.
        and e in (128, 256)
        and (
            (m, t, n, k) == (16, 10, 1280, 2560)
            or (m, t, n, k) == (160, 1, 2560, 640)
        )
    )
    if (
        use_expert_owned_m128
        or use_expert_owned_m64
        or use_expert_owned_m32
        or use_expert_owned_m16
    ):
        # Remote expert slots must remain exact zeros for the later fixed-order
        # reduction.  The sorter and projection use fixed-size device buffers,
        # so this path remains safe under CUDA graph capture.
        out = (
            torch.zeros((m, t, n), dtype=torch.float32, device=x.device)
            if zero_invalid
            else torch.empty((m, t, n), dtype=torch.float32, device=x.device)
        )
        offsets = torch.empty(e + 1, dtype=torch.int32, device=x.device)
        assignments = torch.empty(m * t, dtype=torch.int32, device=x.device)
        _expert_owned_sorter_module(e, m, t).run(
            expert_ids, offsets, assignments
        )
        waves = 4 if (m, t) in ((32, 10), (64, 10), (128, 10)) else 8
        symmetric = (
            os.environ.get("SGLANG_QWEN4_GFX90A_MQ4G128_SYMMETRIC", "0")
            == "1"
        )
        _expert_owned_module(e, m, t, n, k, waves, symmetric).run(
            x, weight, offsets, assignments, out
        )
        return out

    out = torch.empty((m, t, n), dtype=torch.float32, device=x.device)
    if (
        m * t == 10
        and k == 640
        and os.environ.get(
            "SGLANG_QWEN4_GFX90A_MQ4G128_PERSISTENT_SLOTS", "1"
        )
        == "1"
    ):
        _persistent_slots_module(e, m, t, n, k).run(x, weight, expert_ids, out)
    else:
        _indexed_module(e, m, t, n, k).run(x, weight, expert_ids, out)
    return out


def mq4g128_weighted_reduce(
    partials: torch.Tensor, router_weights: torch.Tensor
) -> torch.Tensor:
    m, t, n = partials.shape
    assert m == 1 and partials.dtype == torch.float32 and partials.is_contiguous()
    router_weights = router_weights.float().contiguous()
    out = torch.empty((1, n), dtype=torch.bfloat16, device=partials.device)
    _weighted_reduce_module(t, n).run(partials, router_weights, out)
    return out


def mq4g128_masked_weighted_reduce(
    partials: torch.Tensor,
    router_weights: torch.Tensor,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    m, t, n = partials.shape
    assert partials.dtype == torch.float32 and partials.is_contiguous()
    router_weights = router_weights.float().contiguous()
    expert_ids = expert_ids.to(torch.int32).contiguous()
    out = torch.empty((m, n), dtype=torch.bfloat16, device=partials.device)
    _masked_weighted_reduce_module(m, t, n).run(
        partials, router_weights, expert_ids, out
    )
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
