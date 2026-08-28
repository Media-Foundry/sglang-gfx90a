from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module(
    down_rows: int = 1,
    up_rows: int = 1,
    down_split: int = 1,
    fuse_split_reduce: bool = True,
) -> Module:
    args = make_cpp_args(down_rows, up_rows, down_split, int(fuse_split_reduce))
    return load_jit(
        "gfx90a_qwen_hc_mix_rows",
        *args,
        cuda_files=["hyperconnection/gfx90a_hc_mix.cuh"],
        cuda_wrappers=[
            ("run", f"sglang::Gfx90aQwenHcMix<{args}>::run"),
            (
                "run_with_gate",
                f"sglang::Gfx90aQwenHcMix<{args}>::run_with_gate",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qwen_hc_mix(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    # The model-final mixer has no inject gate and occurs only once per token;
    # keep its exact single-K path.  The production per-block path below owns
    # nearly all HC calls and uses the occupancy split.
    workspace = torch.empty((1, 320), dtype=torch.float32, device=x.device)
    out = torch.empty((1, 2560), dtype=x.dtype, device=x.device)
    rows = int(os.environ.get("SGLANG_QWEN4_GFX90A_HC_ROWS", "1"))
    if rows not in (1, 2, 4, 8):
        raise ValueError("SGLANG_QWEN4_GFX90A_HC_ROWS must be 1, 2, 4, or 8")
    _module(rows, rows, 1).run(x, w_down, w_up, workspace, out)
    return out


def gfx90a_qwen_hc_mix_with_gate(
    x: torch.Tensor,
    w_down: torch.Tensor,
    w_up: torch.Tensor,
    inject_weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    split = int(os.environ.get("SGLANG_QWEN4_GFX90A_HC_DOWN_SPLIT", "4"))
    if split not in (1, 2, 4, 8):
        raise ValueError("SGLANG_QWEN4_GFX90A_HC_DOWN_SPLIT must be 1, 2, 4, or 8")
    fuse_split_reduce = (
        os.environ.get("SGLANG_QWEN4_GFX90A_HC_SPLIT_REDUCE_IN_UP", "1") != "0"
    )
    workspace_rows = split if fuse_split_reduce or split == 1 else split + 1
    workspace = torch.empty(
        (workspace_rows, 320), dtype=torch.float32, device=x.device
    )
    gate_partials = torch.empty((1, 8, 4), dtype=torch.float32, device=x.device)
    out = torch.empty((1, 2560), dtype=x.dtype, device=x.device)
    rows = int(os.environ.get("SGLANG_QWEN4_GFX90A_HC_ROWS", "1"))
    if rows not in (1, 2, 4, 8):
        raise ValueError("SGLANG_QWEN4_GFX90A_HC_ROWS must be 1, 2, 4, or 8")
    _module(rows, rows, split, fuse_split_reduce).run_with_gate(
        x, w_down, w_up, inject_weight, workspace, gate_partials, out
    )
    return out, gate_partials
