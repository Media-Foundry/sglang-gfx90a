from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _module() -> Module:
    configured = os.environ.get("SGLANG_CK_ROOT")
    if configured:
        ck = Path(configured)
    else:
        import aiter

        ck = (
            Path(aiter.__file__).resolve().parent.parent
            / "3rdparty"
            / "composable_kernel"
        )
    return load_jit(
        "gfx90a_ck_hc_mix_m32",
        cuda_files=["hyperconnection/gfx90a_ck_hc_mix_m32.cuh"],
        cuda_wrappers=[
            ("up", "sglang::Gfx90aCkHcMixM32::up"),
            ("epilogue", "sglang::Gfx90aCkHcMixM32::epilogue"),
        ],
        extra_cuda_cflags=["-O3"],
        extra_include_paths=[str(ck / "include"), str(ck / "library" / "include")],
    )


def gfx90a_ck_hc_mix_m32(
    x: torch.Tensor, w_down: torch.Tensor, w_up: torch.Tensor
) -> torch.Tensor:
    assert x.shape == (32, 10240) and x.dtype == torch.bfloat16
    lowrank = F.silu(F.linear(x, w_down) / 4)
    if not (lowrank.is_contiguous() and w_up.is_contiguous()):
        raise RuntimeError(
            "Qwen HC CK M32 requires contiguous operands: "
            f"lowrank={lowrank.shape}/{lowrank.stride()} "
            f"w_up={w_up.shape}/{w_up.stride()}"
        )
    gates = torch.empty_like(x)
    out = torch.empty((32, 2560), dtype=torch.bfloat16, device=x.device)
    module = _module()
    module.up(lowrank, w_up, gates)
    module.epilogue(gates, x, out)
    return out
