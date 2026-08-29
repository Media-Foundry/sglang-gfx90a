from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.cpp_extension import include_paths

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_tp4_m32_ar_mhc_post_oracle_v1",
        cuda_files=["deepseek_v4/gfx90a_tp4_m32_ar_mhc_post_oracle.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aTp4M32ArMhcPostOracle::run"),
            ("run_debug", "sglang::Gfx90aTp4M32ArMhcPostOracle::run_debug"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[
            *include_paths(),
            "/home/pc/pytorch/third_party/aiter/csrc/include",
            "/home/pc/pytorch/third_party/aiter/3rdparty/composable_kernel/include",
        ],
    )


def run(
    fa: int,
    input: torch.Tensor,
    sync_workspace: torch.Tensor,
    residual: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    output: torch.Tensor,
    rms_partials: torch.Tensor,
    reduced_debug: torch.Tensor,
    rank: int,
    *,
    write_reduced: bool = False,
) -> None:
    op = _jit_module().run_debug if write_reduced else _jit_module().run
    op(
        fa,
        input,
        sync_workspace,
        residual,
        post,
        comb,
        output,
        rms_partials,
        reduced_debug,
        rank,
    )
