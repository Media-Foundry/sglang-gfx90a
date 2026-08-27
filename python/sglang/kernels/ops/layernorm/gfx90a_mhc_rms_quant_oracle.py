from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _gfx90a_mhc_rms_quant_oracle_kernel(
    x,
    weight,
    out,
    output_q,
    output_scale,
    eps: tl.constexpr,
    H: tl.constexpr,
    GROUP: tl.constexpr,
):
    token = tl.program_id(0)
    offsets = tl.arange(0, H)
    values = tl.load(x + token * H + offsets).to(tl.float32)
    rms = tl.rsqrt(tl.sum(values * values, axis=0) / H + eps)
    norm_weight = tl.load(weight + offsets).to(tl.float32)
    # The production RMS kernel stores this expression to BF16.  Quantize the
    # explicitly rounded value so the preserved ffn_input and routed-gate
    # activation have exactly the same boundary as the two-launch reference.
    rounded = (values * rms * norm_weight).to(tl.bfloat16)
    tl.store(out + token * H + offsets, rounded)

    grouped = tl.reshape(rounded, (H // GROUP, GROUP)).to(tl.float32)
    absmax = tl.maximum(tl.max(tl.abs(grouped), axis=1), 1.0e-10)
    scale = absmax / 127.0
    quantized = tl.clamp(grouped / scale[:, None], -128.0, 127.0).to(tl.int8)
    tl.store(output_q + token * H + offsets, tl.reshape(quantized, (H,)))
    group_offsets = tl.arange(0, H // GROUP)
    tl.store(output_scale + token * (H // GROUP) + group_offsets, scale)


def gfx90a_mhc_rms_quant_oracle(
    x: torch.Tensor,
    weight: torch.Tensor,
    out: torch.Tensor,
    output_q: torch.Tensor,
    output_scale: torch.Tensor,
    eps: float,
) -> None:
    tokens = x.shape[0]
    if (
        x.shape != (tokens, 4096)
        or weight.shape != (4096,)
        or out.shape != x.shape
        or output_q.shape != x.shape
        or output_scale.shape != (tokens, 128)
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or out.dtype != torch.bfloat16
        or output_q.dtype != torch.int8
        or output_scale.dtype != torch.float32
        or not all(t.is_contiguous() for t in (x, weight, out, output_q, output_scale))
    ):
        raise ValueError("gfx90a MHC RMS+quant oracle expects contiguous Mx4096 tensors")
    _gfx90a_mhc_rms_quant_oracle_kernel[(tokens,)](
        x,
        weight,
        out,
        output_q,
        output_scale,
        eps=eps,
        H=4096,
        GROUP=32,
        num_warps=8,
        num_stages=1,
    )
