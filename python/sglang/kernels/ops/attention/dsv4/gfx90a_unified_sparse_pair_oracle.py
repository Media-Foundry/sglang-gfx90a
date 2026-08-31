from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import include_paths

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


_AITER_ROOT = "/home/pc/pytorch/third_party/aiter"


@cache_once
def _jit_module() -> Module:
    if not torch.version.hip:
        raise RuntimeError("DSV4 pair-query oracle requires ROCm")
    arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "").split(
        ":", 1
    )[0]
    if arch != "gfx90a":
        raise RuntimeError(f"DSV4 pair-query oracle requires gfx90a, got {arch!r}")
    return load_jit(
        "gfx90a_dsv4_unified_sparse_pair_oracle_v1",
        cuda_files=["deepseek_v4/gfx90a_dsv4_unified_sparse_pair_oracle.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aDsv4UnifiedSparsePairOracle::run")
        ],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++20",
            "-DCK_ENABLE_BF16",
            "-DCK_USE_XDL",
        ],
        extra_include_paths=[
            *include_paths(),
            f"{_AITER_ROOT}/3rdparty/composable_kernel/include",
            f"{_AITER_ROOT}/3rdparty/composable_kernel/library/include",
        ],
    )


def preload() -> None:
    """Compile before HIP graph capture and validate the physical architecture."""
    _jit_module()


def run(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    softmax_scale: float,
    compress_ratio: int,
) -> None:
    if compress_ratio != 128:
        raise ValueError(
            f"DSV4 pair-query oracle supports only C128, got C{compress_ratio}"
        )
    _jit_module().run(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        output,
        workspace,
        softmax_scale,
        compress_ratio,
    )


__all__ = ["preload", "run"]
