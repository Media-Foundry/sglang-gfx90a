"""V4.1 TP8 W4A16: CKTile gate/up plus compact, fixed-order HIP down.

Correctness path for the legacy A16W4 scale ABI. Original packed weights and
E8M0 scales stay resident; no 384->512 full-model padding or BF16 weight cache.
"""

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _module():
    return load_jit(
        "gfx90a_dsv41_compact_down_v1",
        cuda_files=["deepseek_v4/gfx90a_dsv41_compact_down.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aDsv41CompactDown::run")],
        extra_cuda_cflags=["-O3", "-std=c++20"],
    )


def compact_down(intermediate, w2, s2, topk_ids, topk_weights, *, logical_k=288, out=None):
    if out is None:
        out = torch.empty((intermediate.shape[0], 5120), dtype=torch.bfloat16, device=intermediate.device)
    partial = torch.empty((intermediate.shape[0], 6, 5120), dtype=torch.float32, device=intermediate.device)
    _module().run(intermediate, w2.view(torch.uint8), s2.view(torch.uint8),
                  topk_ids, topk_weights, partial, out, logical_k)
    return out


def compact_ck_moe(hidden, w13, s13, w2, s2, topk_ids, topk_weights, *, out=None):
    from aiter import ActivationType
    from aiter.fused_moe import cktile_moe_stage1, moe_sorting

    if not (hidden.dtype == torch.bfloat16 and hidden.shape[1] == 5120
            and w13.shape[1:] == (768, 2560) and w2.shape[1:] == (5120, 192)
            and topk_ids.shape == (hidden.shape[0], 6)):
        raise ValueError("V4.1 compact CK requires TP8 H5120/I288 padded384/Top6")
    block_m = 16 if hidden.shape[0] < 2048 else 32 if hidden.shape[0] < 16384 else 64
    sorted_ids, _, expert_ids, valid_ids, moe_out = moe_sorting(
        topk_ids, topk_weights, w13.shape[0], 5120, hidden.dtype, block_m,
        None, None, 0, out)
    intermediate = cktile_moe_stage1(
        hidden, w13, w2, sorted_ids, expert_ids, valid_ids, None, 6,
        block_m=block_m, a1_scale=None, w1_scale=s13.view(torch.float8_e8m0fnu),
        n_pad_zeros=128, activation=ActivationType.Dsv4Silu,
    )
    return compact_down(intermediate, w2, s2, topk_ids, topk_weights, out=moe_out)
