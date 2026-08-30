from __future__ import annotations

import torch

from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args


def repack_fp4_to_signed_int5(weight: torch.Tensor, *, bitplane: bool = False) -> torch.Tensor:
    """Exact load-time oracle repack: 32 E2M1 nibbles -> five uint32 words."""
    if weight.dtype != torch.uint8 or weight.shape[-1] % 16:
        raise ValueError("packed FP4 weight must be uint8 with last dim % 16 == 0")
    nibble = torch.stack((weight & 15, weight >> 4), dim=-1).flatten(-2)
    mag = nibble & 7
    code = mag + (mag > 4) + (mag > 5) + 3 * (mag > 6)
    signed = torch.where((nibble & 8) != 0, -code, code).to(torch.int32) & 31
    groups = signed.reshape(*signed.shape[:-1], -1, 32)
    if bitplane:
        low = groups & 15
        out = torch.zeros((*groups.shape[:-1], 5), dtype=torch.int64, device=weight.device)
        for word in range(4):
            for j in range(8):
                out[..., word] |= low[..., word * 8 + j].to(torch.int64) << (j * 4)
        for j in range(32):
            out[..., 4] |= ((groups[..., j] >> 4).to(torch.int64) & 1) << j
        return out.to(torch.uint32).contiguous()
    out = torch.zeros((*groups.shape[:-1], 5), dtype=torch.int64, device=weight.device)
    for j in range(32):
        bit = j * 5
        word, shift = divmod(bit, 32)
        value = groups[..., j].to(torch.int64)
        out[..., word] |= (value << shift) & 0xFFFFFFFF
        if shift > 27:
            out[..., word + 1] |= value >> (32 - shift)
    return out.to(torch.uint32).contiguous()


@cache_once
def _jit_down(e: int, m: int, t: int, n: int, k: int, bitplane: bool) -> object:
    args = make_cpp_args(e, m, t, n, k, 4, 2, 8, 832, bitplane)
    return load_jit(
        "gfx90a_fp4_int5_down_oracle",
        *args,
        cuda_files=[
            "deepseek_v4/gfx90a_fp4_expert_gemv.cuh",
            "deepseek_v4/gfx90a_fp4_int5_repack_oracle.cuh",
        ],
        cuda_wrappers=[(
            "run_partial",
            f"sglang::Gfx90aFp4Int5DownOracleKernel<{args}>::run_partial",
        )],
        extra_cuda_cflags=["-O3", "-save-temps"],
    )


def run_int5_down_partial(
    xq: torch.Tensor, x_scale: torch.Tensor, int5_weight: torch.Tensor,
    weight_scale: torch.Tensor, sorted_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor, num_valid_ids: torch.Tensor,
    topk_weights: torch.Tensor, partial: torch.Tensor, *, bitplane: bool = False,
) -> None:
    e, n, groups, words = int5_weight.shape
    assert words == 5
    m, t, k = xq.shape
    assert groups == k // 32
    _jit_down(e, m, t, n, k, bitplane).run_partial(
        xq, x_scale, int5_weight, weight_scale.view(torch.uint8).reshape(e, n, groups),
        sorted_ids, sorted_expert_ids, num_valid_ids, topk_weights, partial,
    )
