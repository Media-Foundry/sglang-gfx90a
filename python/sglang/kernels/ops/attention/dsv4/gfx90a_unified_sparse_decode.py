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
    return load_jit(
        "gfx90a_dsv4_unified_sparse_decode_v3",
        cuda_files=["deepseek_v4/gfx90a_dsv4_unified_sparse_decode.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aDsv4UnifiedSparseDecode::run")],
        extra_cuda_cflags=["-O3", "-std=c++20", "-DCK_ENABLE_BF16", "-DCK_USE_XDL"],
        extra_include_paths=[
            *include_paths(),
            f"{_AITER_ROOT}/3rdparty/composable_kernel/include",
            f"{_AITER_ROOT}/3rdparty/composable_kernel/library/include",
        ],
    )


def run(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices: torch.Tensor,
    kv_indptr: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    workspace: torch.Tensor,
    softmax_scale: float,
) -> None:
    _jit_module().run(
        q,
        unified_kv,
        kv_indices,
        kv_indptr,
        attn_sink,
        output,
        workspace,
        softmax_scale,
    )


def workspace_size_bytes(tokens: int = 64, heads: int = 16, splits: int = 2) -> int:
    return tokens * splits * heads * (512 + 2) * 4
