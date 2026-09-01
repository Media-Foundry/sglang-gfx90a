from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gate_up(
    e: int,
    m: int,
    t: int,
    ge: int,
    i: int,
    k: int,
    blocks: int,
    slot_begin: int,
    slot_end: int,
    rows: int,
    waves: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, ge, i, k, rows, waves, blocks, slot_begin, slot_end
    )
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
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    assignments: int,
    rows: int,
    waves: int,
    blocks: int,
    prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, waves, blocks, prepacked
    )
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
def _jit_gate_up_grouped_dpp(
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    assignments: int,
    rows: int,
    waves: int,
    blocks: int,
    prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_up_grouped_dpp",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4ExpertGateUpGroupedDppKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gate_up_grouped_dpp_fp16_accum(
    e: int, m: int, t: int, i: int, k: int,
    assignments: int, rows: int, waves: int, blocks: int, prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_up_grouped_dpp_fp16_accum",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4ExpertGateUpGroupedDppFp16AccumKernel<{args}>::run",
            )
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gate_up_mfma32(
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    blocks: int,
    split: int,
    broadcast_scales: int,
    assignments: int,
    preshuffled: bool,
) -> Module:
    args = make_cpp_args(
        e, m, t, i, k, blocks, split, broadcast_scales, assignments,
        preshuffled,
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_up_mfma32",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ExpertGateUpMfma32Kernel<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_mfma64_expert_persistent_gate(
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    blocks: int,
    split: int,
    assignments: int,
) -> Module:
    args = make_cpp_args(e, m, t, i, k, blocks, split, assignments)
    return load_jit(
        "gfx90a_fp4_mfma64_expert_persistent_gate",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_mfma64_expert_persistent_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "build_runs",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentGateOracle<{args}>::build_runs",
            ),
            (
                "run",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentGateOracle<{args}>::run",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down(
    e: int,
    m: int,
    t: int,
    ge: int,
    n: int,
    k: int,
    blocks: int,
    slot_begin: int,
    slot_end: int,
    rows: int,
    waves: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, ge, n, k, rows, waves, blocks, slot_begin, slot_end
    )
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


@cache_once
def _jit_down_grouped(
    e: int,
    m: int,
    t: int,
    n: int,
    k: int,
    assignments: int,
    rows: int,
    waves: int,
    blocks: int,
    prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, n, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_down_grouped",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            (
                "run",
                f"sglang::Gfx90aFp4ExpertDownGroupedKernel<{args}>::run",
            ),
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownGroupedKernel<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownGroupedKernel<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down_grouped_row_prefetch(
    e: int,
    m: int,
    t: int,
    n: int,
    k: int,
    assignments: int,
    waves: int,
    blocks: int,
    prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, n, k, assignments, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_down_row_prefetch",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_down_row_prefetch_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down_grouped_row_prefetch_logical_scale(
    e: int, m: int, t: int, n: int, k: int, assignments: int,
    waves: int, blocks: int, prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, n, k, assignments, waves, blocks, prepacked, True
    )
    return load_jit(
        "gfx90a_fp4_expert_down_row_prefetch_logical_scale",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_down_row_prefetch_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_gate_up_grouped_row_prefetch(
    e: int, m: int, t: int, i: int, k: int, assignments: int,
    rows: int, waves: int, blocks: int, prepacked: int,
) -> Module:
    args = make_cpp_args(
        e, m, t, i, k, assignments, rows, waves, blocks, prepacked
    )
    return load_jit(
        "gfx90a_fp4_expert_gate_row_prefetch",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh"
        ],
        cuda_wrappers=[(
            "run",
            f"sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{args}>::run",
        )],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_down_mfma32(
    e: int,
    m: int,
    t: int,
    n: int,
    k: int,
    blocks: int,
    split: int,
    broadcast_scales: int,
    assignments: int,
    preshuffled: bool,
) -> Module:
    args = make_cpp_args(
        e, m, t, n, k, blocks, split, broadcast_scales, assignments,
        preshuffled,
    )
    return load_jit(
        "gfx90a_fp4_expert_down_mfma32",
        *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gemv.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aFp4ExpertDownMfma32Kernel<{args}>::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def _jit_mfma64_expert_persistent_down(
    e: int,
    m: int,
    t: int,
    n: int,
    k: int,
    blocks: int,
    split: int,
    assignments: int,
) -> Module:
    args = make_cpp_args(e, m, t, n, k, blocks, split, assignments)
    return load_jit(
        "gfx90a_fp4_mfma64_expert_persistent_down",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_mfma64_expert_persistent_oracle.cuh"
        ],
        cuda_wrappers=[
            (
                "run_partial",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentDownOracle<{args}>::run_partial",
            ),
            (
                "reduce",
                f"sglang::Gfx90aFp4Mfma64ExpertPersistentDownOracle<{args}>::reduce",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


_mfma64_expert_run_workspaces: dict[
    tuple[int, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
] = {}


def gfx90a_fp4_mfma64_expert_runs(
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    *,
    e: int,
    m: int,
    t: int,
    i: int,
    k: int,
    assignments: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device_index = sorted_expert_ids.device.index
    key = (device_index if device_index is not None else 0, e)
    workspace = _mfma64_expert_run_workspaces.get(key)
    if workspace is None:
        workspace = (
            torch.empty(e, dtype=torch.int32, device=sorted_expert_ids.device),
            torch.empty(e, dtype=torch.int32, device=sorted_expert_ids.device),
            torch.empty(e, dtype=torch.int32, device=sorted_expert_ids.device),
            torch.full(
                (1,), e, dtype=torch.int32, device=sorted_expert_ids.device
            ),
        )
        _mfma64_expert_run_workspaces[key] = workspace
    active_experts, block_starts, block_counts, num_active = workspace
    _jit_mfma64_expert_persistent_gate(
        e, m, t, i, k, 832, 4, assignments
    ).build_runs(
        sorted_expert_ids,
        num_valid_ids,
        active_experts,
        block_starts,
        block_counts,
    )
    return workspace


def gfx90a_fp4_expert_gate_up_mfma64_persistent(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_runs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    topk: int,
    limit: float,
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = xq.shape
    i = two_i // 2
    assert (e, m, topk, i, k) == (256, 4608, 6, 512, 4096)
    assert packed_k * 2 == k
    out = torch.empty((m, topk, i), dtype=torch.bfloat16, device=xq.device)
    active_experts, block_starts, block_counts, num_active = expert_runs
    _jit_mfma64_expert_persistent_gate(e, m, topk, i, k, 832, 4, 64).run(
        xq,
        x_scale,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, two_i, k // 32),
        sorted_ids,
        active_experts,
        block_starts,
        block_counts,
        num_active,
        out,
        float(limit),
    )
    return out


def gfx90a_fp4_expert_down_mfma64_persistent(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    expert_runs: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    topk_weights: torch.Tensor,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, topk, k = xq.shape
    assert (e, m, topk, n, k) == (256, 4608, 6, 4096, 512)
    assert packed_k * 2 == k
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    partial = torch.empty((m, topk, n), dtype=torch.float32, device=xq.device)
    active_experts, block_starts, block_counts, num_active = expert_runs
    module = _jit_mfma64_expert_persistent_down(
        e, m, topk, n, k, 416, 2, 64
    )
    module.run_partial(
        xq,
        x_scale,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, n, k // 32),
        sorted_ids,
        active_experts,
        block_starts,
        block_counts,
        num_active,
        topk_weights,
        partial,
    )
    module.reduce(partial, out)
    return out


def gfx90a_fp4_expert_gate_up(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    expert_ids: torch.Tensor,
    expert_mask: torch.Tensor | None,
    live_count: torch.Tensor | None,
    limit: float,
    prequant: tuple[torch.Tensor, torch.Tensor] | None = None,
    blocks: int = 208,
    slot_begin: int = 0,
    slot_end: int | None = None,
    rows: int = 2,
    waves: int = 8,
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = x.shape
    i, t = two_i // 2, expert_ids.shape[1]
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    out = torch.empty((m, t, i), dtype=torch.bfloat16, device=x.device)
    slot_end = t if slot_end is None else slot_end
    assert 0 <= slot_begin < slot_end <= t
    kernel = _jit_gate_up(
        e, m, t, ge, i, k, blocks, slot_begin, slot_end, rows, waves
    )
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
    rows: int = 2,
    waves: int = 8,
    blocks: int = 208,
    prepacked_weight: torch.Tensor | None = None,
    use_lds_lut: bool = False,
    use_dpp_reduction: bool = False,
    use_row_prefetch: bool = False,
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
    kernel_weight = weight if prepacked_weight is None else prepacked_weight
    if prepacked_weight is not None:
        assert prepacked_weight.shape == (e, two_i, k)
        assert (
            prepacked_weight.dtype == torch.int8
            and prepacked_weight.is_contiguous()
        )
    assert not (prepacked_weight is not None and use_lds_lut)
    weight_mode = 1 if prepacked_weight is not None else (2 if use_lds_lut else 0)
    if use_row_prefetch:
        assert e == 256 and m in (32, 51, 64)
        assert (topk, i, k) == (6, 512, 4096)
        assert (assignments, rows, waves, weight_mode) == (4, 2, 8, 2)
        assert (
            (m == 51 and blocks == 1664)
            or (m in (32, 64) and blocks == 2080)
        )
    elif use_dpp_reduction:
        assert e == 256 and m in (32, 51, 64)
        assert (topk, i, k) == (6, 512, 4096)
        assert (assignments, rows, waves, weight_mode) == (4, 2, 8, 2)
        assert (
            (m == 51 and blocks == 1664)
            or (m in (32, 64) and blocks == 2080)
        )
    gate_module = (
        _jit_gate_up_grouped_row_prefetch
        if use_row_prefetch
        else _jit_gate_up_grouped_dpp
        if use_dpp_reduction
        else _jit_gate_up_grouped
    )
    gate_module(
        e,
        m,
        topk,
        i,
        k,
        assignments,
        rows,
        waves,
        blocks,
        weight_mode,
    ).run(
        xq,
        x_scale,
        (
            kernel_weight
            if prepacked_weight is not None
            else kernel_weight.view(torch.uint8)
        ),
        weight_scale.view(torch.uint8).reshape(e, two_i, k // 32),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        out,
        float(limit),
    )
    return out


def gfx90a_fp4_expert_gate_up_mfma32(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk: int,
    limit: float,
    blocks: int = 416,
    split: int = 4,
    broadcast_scales: int = 0,
    assignments: int = 32,
    preshuffled: bool = False,
) -> torch.Tensor:
    e, two_i, packed_k = weight.shape
    m, k = xq.shape
    i = two_i // 2
    assert packed_k * 2 == k and i % 16 == 0
    out = torch.empty((m, topk, i), dtype=torch.bfloat16, device=xq.device)
    assert split in (2, 4, 8)
    assert broadcast_scales in (0, 1)
    assert assignments in (32, 64)
    _jit_gate_up_mfma32(
        e, m, topk, i, k, blocks, split, broadcast_scales, assignments,
        preshuffled,
    ).run(
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


def gfx90a_fp4_expert_down_grouped(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor | None = None,
    assignments: int = 2,
    rows: int = 2,
    waves: int = 8,
    blocks: int = 208,
    prepacked_weight: torch.Tensor | None = None,
    use_lds_lut: bool = False,
    zero_partial: bool = False,
    use_row_prefetch: bool = False,
    use_logical_scale: bool = False,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, topk, k = xq.shape
    assert packed_k * 2 == k
    assert xq.dtype == torch.int8 and xq.is_contiguous()
    assert x_scale.shape == (m, topk, k // 32)
    assert x_scale.dtype == torch.float32 and x_scale.is_contiguous()
    assert sorted_ids.dtype == torch.int32 and sorted_ids.is_contiguous()
    assert sorted_expert_ids.dtype == torch.int32
    assert num_valid_ids.shape == (2,) and num_valid_ids.dtype == torch.int32
    assert topk_weights.shape == (m, topk)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    # Split-MoE masks the slots owned by the other DP replica before sorting.
    # Those slots are intentionally never written by the grouped down kernel,
    # while the legacy final reduction still visits all top-k slots.
    partial = (
        torch.zeros((m, topk, n), dtype=torch.float32, device=xq.device)
        if zero_partial
        else torch.empty((m, topk, n), dtype=torch.float32, device=xq.device)
    )
    kernel_weight = weight if prepacked_weight is None else prepacked_weight
    if prepacked_weight is not None:
        assert prepacked_weight.shape == (e, n, k)
        assert (
            prepacked_weight.dtype == torch.int8
            and prepacked_weight.is_contiguous()
        )
    assert not (prepacked_weight is not None and use_lds_lut)
    weight_mode = 1 if prepacked_weight is not None else (2 if use_lds_lut else 0)
    if use_row_prefetch:
        assert e == 256 and m in (32, 51, 64)
        assert (topk, n, k) == (6, 4096, 512)
        assert (assignments, rows, blocks, weight_mode) == (4, 2, 832, 2)
        assert waves == 8 or (m in (51, 64) and waves == 4)
        assert prepacked_weight is None and use_lds_lut
        if use_logical_scale:
            assert weight_scale.shape == (256, 4096, 16)
            assert weight_scale.dtype == torch.uint8 and weight_scale.is_contiguous()
        module_loader = (
            _jit_down_grouped_row_prefetch_logical_scale
            if use_logical_scale
            else _jit_down_grouped_row_prefetch
        )
        module = module_loader(
            e,
            m,
            topk,
            n,
            k,
            assignments,
            waves,
            blocks,
            weight_mode,
        )
        module.run_partial(
            xq,
            x_scale,
            kernel_weight.view(torch.uint8),
            weight_scale.view(torch.uint8).reshape(e, n, k // 32),
            sorted_ids,
            sorted_expert_ids,
            num_valid_ids,
            topk_weights,
            partial,
        )
        module.reduce(partial, out)
        return out
    _jit_down_grouped(
        e,
        m,
        topk,
        n,
        k,
        assignments,
        rows,
        waves,
        blocks,
        weight_mode,
    ).run(
        xq,
        x_scale,
        (
            kernel_weight
            if prepacked_weight is not None
            else kernel_weight.view(torch.uint8)
        ),
        weight_scale.view(torch.uint8).reshape(e, n, k // 32),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk_weights,
        partial,
        out,
    )
    return out


def gfx90a_fp4_expert_down_mfma32(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    out: torch.Tensor | None = None,
    blocks: int = 312,
    split: int = 4,
    broadcast_scales: int = 0,
    assignments: int = 32,
    preshuffled: bool = False,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, topk, k = xq.shape
    assert packed_k * 2 == k and n % 16 == 0
    partial = torch.empty((m, topk, n), dtype=torch.float32, device=xq.device)
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=xq.device)
    assert split in (2, 4, 8)
    assert broadcast_scales in (0, 1)
    assert assignments in (32, 64)
    _jit_down_mfma32(
        e, m, topk, n, k, blocks, split, broadcast_scales, assignments,
        preshuffled,
    ).run(
        xq,
        x_scale,
        weight.view(torch.uint8),
        weight_scale.view(torch.uint8).reshape(e, n, k // 32),
        sorted_ids,
        sorted_expert_ids,
        num_valid_ids,
        topk_weights,
        partial,
        out,
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
    blocks: int = 208,
    slot_begin: int = 0,
    slot_end: int | None = None,
    rows: int = 2,
    waves: int = 8,
) -> torch.Tensor:
    e, n, packed_k = weight.shape
    m, t, k = x.shape
    ge = e if expert_mask is None else expert_mask.numel()
    assert packed_k * 2 == k
    if out is None:
        out = torch.empty((m, n), dtype=torch.bfloat16, device=x.device)
    slot_end = t if slot_end is None else slot_end
    assert 0 <= slot_begin < slot_end <= t
    kernel = _jit_down(
        e, m, t, ge, n, k, blocks, slot_begin, slot_end, rows, waves
    )
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
