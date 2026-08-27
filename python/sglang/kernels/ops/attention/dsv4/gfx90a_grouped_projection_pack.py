from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_grouped_projection_pack_m32_n520",
        cuda_files=["deepseek_v4/gfx90a_grouped_projection_pack.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aGroupedProjectionPackKernel::run")
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_grouped_projection_pack(
    grouped: torch.Tensor, rank: int, out: torch.Tensor | None = None
) -> torch.Tensor:
    if out is None:
        out = torch.empty((32, 520), dtype=torch.bfloat16, device=grouped.device)
    _jit_module().run(grouped, out, rank)
    return out
