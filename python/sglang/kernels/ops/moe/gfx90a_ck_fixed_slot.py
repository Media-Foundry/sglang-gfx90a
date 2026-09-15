"""Opt-in deterministic CK stage-2 reduction for original V4 large prefill.

No CK binary or GEMM arithmetic is changed. Each Top-6 assignment becomes a
virtual token with a unique FP32 output, followed by a fixed-order HIP sum.
This uses six times the original FP32 accumulation scratch; default remains off.
"""
import logging
import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def _announce():
    logging.getLogger(__name__).warning(
        "DSV4 large-prefill CK fixed-slot FP32 reduction selected (Top6, extra scratch)"
    )


@cache_once
def fixed_slot_module():
    return load_jit(
        "gfx90a_ck_fixed_slot",
        cuda_files=["deepseek_v4/gfx90a_ck_fixed_slot.cuh"],
        cuda_wrappers=[
            ("remap", "sglang::Gfx90aCkFixedSlot::remap"),
            ("reduce", "sglang::Gfx90aCkFixedSlot::reduce"),
            ("reduce_float", "sglang::Gfx90aCkFixedSlot::reduce_float"),
        ],
        extra_cuda_cflags=["-O3", "-fno-fast-math", "-ffp-contract=off"],
    )


def ck_fixed_slot_stage2(ck_entry, inter, w1, w2, sorted_ids, sorted_experts,
                         valid, out, topk, kernel_name, w2_scale, a2_scale,
                         block_m, sorted_weights, quant_type, activation, nt):
    m, t, k = inter.shape
    if not (8192 <= m <= 36864 and t == topk == 6 and k in (256, 512)):
        raise ValueError("fixed-slot CK requires large-prefill DSV4 Top-6 shape")
    if inter.dtype != torch.bfloat16 or not inter.is_contiguous():
        raise ValueError("fixed-slot CK requires contiguous BF16 intermediate")
    if out.shape != (m, 4096) or out.dtype != torch.bfloat16:
        raise ValueError("fixed-slot CK requires BF16 [M,4096] output")
    if w2_scale is not None or a2_scale is not None or quant_type != 0:
        raise ValueError("fixed-slot CK requires materialized unquantized BF16 inputs")
    _announce()
    mod = fixed_slot_module()
    remapped = torch.empty_like(sorted_ids)
    mod.remap(sorted_ids, valid, remapped, m)
    manifest = os.getenv('SGLANG_DSV4_DEBUG_CK_UNIQUE_MANIFEST')
    if manifest:
        from sglang.kernels.ops.debug.dsv4_ck_unique_store import eligible, load_verified
        from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
        if not mix_pair_active() or not eligible(m, t, k, block_m, kernel_name):
            raise ValueError('unique CK experiment requires original-V4 TP8 native large-prefill scope')
        if sorted_weights is None:
            raise ValueError('unique CK experiment requires complete routed weights')
        partial = torch.empty((m * 6, 4096), device=out.device, dtype=torch.float32)
        load_verified(manifest).stage2(inter.view(m * 6, 1, k), w2, remapped,
            sorted_experts, valid, sorted_weights, partial)
    else:
        partial = torch.zeros((m * 6, 4096), device=out.device, dtype=torch.float32)
        ck_entry(inter.view(m * 6, 1, k), w1, w2, remapped, sorted_experts,
                 valid, partial, 1, kernel_name, None, None, block_m, sorted_weights,
                 quant_type, activation, nt)
    mod.reduce(partial.view(m, 6, 4096), out)
    return out
