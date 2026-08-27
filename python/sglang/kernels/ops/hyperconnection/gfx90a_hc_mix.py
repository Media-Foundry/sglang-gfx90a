from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    return load_jit(
        "gfx90a_qwen_hc_mix",
        cuda_files=["hyperconnection/gfx90a_hc_mix.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aQwenHcMix::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_qwen_hc_mix(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    workspace = torch.empty((1, 320), dtype=torch.float32, device=x.device)
    out = torch.empty((1, 2560), dtype=x.dtype, device=x.device)
    _module().run(x, w_down, w_up, workspace, out)
    return out
