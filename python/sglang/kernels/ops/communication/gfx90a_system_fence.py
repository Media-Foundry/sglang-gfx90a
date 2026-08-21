from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_system_fence_module() -> Module:
    return load_jit(
        "gfx90a_system_fence",
        cuda_files=["distributed/gfx90a_system_fence.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aSystemFenceKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_system_fence(anchor: torch.Tensor) -> None:
    """Publish prior writes at system scope on the current HIP stream."""
    _jit_gfx90a_system_fence_module().run(anchor.view(-1))
