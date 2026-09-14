from __future__ import annotations

import os
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
        cuda_wrappers=[("run", "sglang::Gfx90aRealtimeMarkerKernel::run"),
                       ("wall_clock_khz", "sglang::Gfx90aRealtimeMarkerKernel::wall_clock_khz")],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_realtime_marker(output: torch.Tensor, slot: int) -> None:
    if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR"):
        from sglang.kernels.ops.debug.dsv4_prefill_markers import active, select_detail_row

        if not active():
            return
        select_detail_row(output, slot)
    module = _jit_marker()
    # Isolate decode graph instrumentation from eager prefill schedules.
    # Resolve the module even on warmup so first use need not load it in capture.
    if os.getenv("SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY", "0") == "1":
        if not torch.cuda.is_current_stream_capturing():
            return
    module.run(output, slot)
