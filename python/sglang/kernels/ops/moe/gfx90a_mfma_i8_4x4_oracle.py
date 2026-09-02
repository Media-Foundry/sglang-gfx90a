from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_module() -> Module:
    return load_jit(
        "gfx90a_mfma_i8_4x4_oracle",
        cuda_files=["deepseek_v4/gfx90a_mfma_i8_4x4_oracle.cuh"],
        cuda_wrappers=[
            ("probe", "sglang::Gfx90aMfmaI8_4x4ProbeKernel::run"),
            (
                "a4n4k32",
                "sglang::Gfx90aMfmaI8A4N4K32OracleKernel::run",
            ),
            (
                "m32n32k32",
                "sglang::Gfx90aMfmaI8M32N32K32OracleKernel<true,true>::run",
            ),
            (
                "m32n32k32_mfma",
                "sglang::Gfx90aMfmaI8M32N32K32OracleKernel<true,false>::run",
            ),
            (
                "m32n32k32_sdot",
                "sglang::Gfx90aMfmaI8M32N32K32OracleKernel<false,true>::run",
            ),
        ],
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_mfma_i8_4x4_probe(
    a: torch.Tensor,
    b: torch.Tensor,
    *,
    cbsz: int = 0,
    abid: int = 0,
    blgp: int = 0,
) -> torch.Tensor:
    out = torch.empty((64, 4), dtype=torch.int32, device=a.device)
    _jit_module().probe(a, b, out, cbsz, abid, blgp)
    return out


def gfx90a_mfma_i8_a4n4k32_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    x_scale: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    outputs = (
        torch.empty((4, 4), dtype=torch.int32, device=x.device),
        torch.empty((4, 4), dtype=torch.int32, device=x.device),
        torch.empty((4, 4), dtype=torch.float32, device=x.device),
        torch.empty((4, 4), dtype=torch.float32, device=x.device),
    )
    _jit_module().a4n4k32(x, weight, x_scale, weight_scale, *outputs)
    return outputs


def gfx90a_mfma_i8_m32n32k32_oracle(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    mfma_out = torch.empty((32, 32), dtype=torch.int32, device=x.device)
    sdot_out = torch.empty_like(mfma_out)
    _jit_module().m32n32k32(x, weight, mfma_out, sdot_out)
    return mfma_out, sdot_out


@cache_once
def _jit_gate_tile_module() -> Module:
    wrappers = [
        (
            "reference_check",
            "sglang::Gfx90aSdotI8GateTileReferenceKernel<true>::run",
        ),
        (
            "reference_timed",
            "sglang::Gfx90aSdotI8GateTileReferenceKernel<false>::run",
        ),
    ]
    for split in (1, 2, 4, 8):
        wrappers.extend(
            (
                (
                    f"mfma_split{split}_check",
                    f"sglang::Gfx90aMfmaI8GateTileOracleKernel<{split},true>::run",
                ),
                (
                    f"mfma_split{split}_timed",
                    f"sglang::Gfx90aMfmaI8GateTileOracleKernel<{split},false>::run",
                ),
            )
        )
    return load_jit(
        "gfx90a_mfma_i8_gate_tile_oracle",
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_mfma_i8_gate_tile_oracle.cuh",
        ],
        cuda_wrappers=wrappers,
        extra_cuda_cflags=["-O3"],
    )


def gfx90a_mfma_i8_gate_tile_oracle(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    *,
    split: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if split not in (1, 2, 4, 8):
        raise ValueError(f"unsupported split={split}")
    out = torch.empty((4, 64), dtype=torch.float32, device=xq.device)
    group_int = torch.empty((128, 4, 64), dtype=torch.int32, device=xq.device)
    getattr(_jit_gate_tile_module(), f"mfma_split{split}_check")(
        xq, x_scale, weight, weight_scale, out, group_int
    )
    return out, group_int


def gfx90a_sdot_i8_gate_tile_reference(
    xq: torch.Tensor,
    x_scale: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty((4, 64), dtype=torch.float32, device=xq.device)
    group_int = torch.empty((128, 4, 64), dtype=torch.int32, device=xq.device)
    _jit_gate_tile_module().reference_check(
        xq, x_scale, weight, weight_scale, out, group_int
    )
    return out, group_int
