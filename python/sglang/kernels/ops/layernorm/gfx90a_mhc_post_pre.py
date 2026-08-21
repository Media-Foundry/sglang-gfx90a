from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.kernels.jit.utils import cache_once, load_jit
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_gfx90a_mhc_post_pre_module() -> Module:
    fast_math = envs.SGLANG_DSV4_GFX90A_MHC_FAST_MATH.get()
    return load_jit(
        f"gfx90a_mhc_post_pre_wave64_v3_{'fast' if fast_math else 'precise'}",
        cuda_files=["deepseek_v4/gfx90a_mhc_post_pre.cuh"],
        cuda_wrappers=[
            ("run", "sglang::Gfx90aMhcPostPreKernel::run"),
            ("finish", "sglang::Gfx90aMhcFinishKernel::run"),
        ],
        extra_cuda_cflags=["-O3"] + (["-ffast-math"] if fast_math else []),
    )


def preload_gfx90a_mhc_post_pre() -> None:
    _jit_gfx90a_mhc_post_pre_module()


def gfx90a_mhc_post_pre(
    x: torch.Tensor,
    residual: torch.Tensor,
    previous_post: torch.Tensor,
    previous_comb: torch.Tensor,
    fn: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    rms_eps: float,
    sinkhorn_eps: float,
    post_multiplier: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if (
        not torch.version.hip
        or x.ndim != 2
        or x.shape[1] != 4096
        or residual.shape != (x.shape[0], 4, 4096)
        or previous_post.shape != (x.shape[0], 4)
        or previous_comb.shape != (x.shape[0], 4, 4)
        or fn.shape != (24, 16384)
        or hc_scale.shape != (3,)
        or hc_base.shape != (24,)
        or norm_weight.shape != (4096,)
        or x.dtype != torch.bfloat16
        or residual.dtype != torch.bfloat16
        or previous_post.dtype != torch.float32
        or previous_comb.dtype != torch.float32
        or fn.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
        or norm_weight.dtype != torch.bfloat16
        or not all(
            tensor.is_contiguous()
            for tensor in (
                x,
                residual,
                previous_post,
                previous_comb,
                fn,
                hc_scale,
                hc_base,
                norm_weight,
            )
        )
        or getattr(torch.cuda.get_device_properties(x.device), "gcnArchName", "").split(
            ":", 1
        )[0]
        != "gfx90a"
    ):
        return None
    residual_out = torch.empty_like(residual)
    post_out = torch.empty_like(previous_post)
    comb_out = torch.empty_like(previous_comb)
    layer_input_out = torch.empty_like(x)
    _jit_gfx90a_mhc_post_pre_module().run(
        x,
        residual,
        previous_post,
        previous_comb,
        fn,
        hc_scale,
        hc_base,
        norm_weight,
        residual_out,
        post_out,
        comb_out,
        layer_input_out,
        float(rms_eps),
        float(sinkhorn_eps),
        float(post_multiplier),
        float(norm_eps),
    )
    return residual_out, post_out, comb_out, layer_input_out


def gfx90a_mhc_finish(
    residual: torch.Tensor,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    norm_weight: torch.Tensor,
    sinkhorn_eps: float,
    post_multiplier: float,
    norm_eps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    num_tokens = residual.shape[0]
    if (
        not torch.version.hip
        or residual.shape != (num_tokens, 4, 4096)
        or mixes.shape != (num_tokens, 24)
        or hc_scale.shape != (3,)
        or hc_base.shape != (24,)
        or norm_weight.shape != (4096,)
        or residual.dtype != torch.bfloat16
        or mixes.dtype != torch.float32
        or hc_scale.dtype != torch.float32
        or hc_base.dtype != torch.float32
        or norm_weight.dtype != torch.bfloat16
        or not all(
            tensor.is_contiguous()
            for tensor in (residual, mixes, hc_scale, hc_base, norm_weight)
        )
        or getattr(
            torch.cuda.get_device_properties(residual.device), "gcnArchName", ""
        ).split(":", 1)[0]
        != "gfx90a"
    ):
        return None
    post_out = torch.empty((num_tokens, 4), dtype=torch.float32, device=residual.device)
    comb_out = torch.empty(
        (num_tokens, 4, 4), dtype=torch.float32, device=residual.device
    )
    layer_input_out = torch.empty(
        (num_tokens, 4096), dtype=torch.bfloat16, device=residual.device
    )
    _jit_gfx90a_mhc_post_pre_module().finish(
        residual,
        mixes,
        hc_scale,
        hc_base,
        norm_weight,
        post_out,
        comb_out,
        layer_input_out,
        float(sinkhorn_eps),
        float(post_multiplier),
        float(norm_eps),
    )
    return post_out, comb_out, layer_input_out
