from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_marker() -> Module:
    return load_jit(
        "gfx90a_realtime_marker",
        cuda_files=["debug/gfx90a_realtime_marker.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aRealtimeMarkerKernel::run")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_realtime_marker(output: torch.Tensor, slot: int) -> None:
    _jit_marker().run(output, slot)
