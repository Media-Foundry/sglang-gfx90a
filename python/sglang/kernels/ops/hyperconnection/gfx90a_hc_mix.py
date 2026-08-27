from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(down_rows: int = 1, up_rows: int = 1) -> Module:
    args = make_cpp_args(down_rows, up_rows)
    return load_jit(
        "gfx90a_qwen_hc_mix_rows",
        *args,
        cuda_files=["hyperconnection/gfx90a_hc_mix.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aQwenHcMix<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qwen_hc_mix(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    workspace = torch.empty((1, 320), dtype=torch.float32, device=x.device)
    out = torch.empty((1, 2560), dtype=x.dtype, device=x.device)
    rows = int(os.environ.get("SGLANG_QWEN4_GFX90A_HC_ROWS", "1"))
    if rows not in (1, 2, 4, 8):
        raise ValueError("SGLANG_QWEN4_GFX90A_HC_ROWS must be 1, 2, 4, or 8")
    _module(rows, rows).run(x, w_down, w_up, workspace, out)
    return out
