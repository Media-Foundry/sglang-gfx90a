"""Fixed-reduction BF16 wo_a for V4.1 HIP numerical validation.

Reuse SGLang's batch-invariant GEMM implementation with a small wave64 tile.
No global batch-invariant mode, weight conversion/cache, or M-based tactic
selection is enabled. Identical rows use the same FP32 accumulation order.
"""

import torch
import triton


def wo_a_bf16_invariant(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if x.ndim != 3 or weight.ndim != 3 or x.shape[1] != weight.shape[0] or x.shape[2] != weight.shape[2]:
        raise ValueError("Expected x[T,G,D] and weight[G,R,D]")
    if x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or x.device != weight.device:
        raise ValueError("wo_a requires BF16 operands on the same device")
    if x.device.type != "cuda" or torch.version.hip is None:
        raise ValueError("This wo_a entry is HIP-only")
    if x.shape[1] not in (1, 2) or weight.shape[1:] != (1024, 4096) or x.stride(2) != 1 or not weight.is_contiguous():
        raise ValueError("Validated contract is TP8/TP4 G1/G2, R1024, D4096 with contiguous weights")
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_kernel_persistent

    t, g, d = x.shape
    r = weight.shape[1]
    out = torch.empty((t, g, r), device=x.device, dtype=x.dtype)
    if t == 0:
        return out
    # Fixed geometry for decode AND extend. Changing M must not select a new
    # reduction tree just before wo_b's discontinuous FP8 activation quant.
    sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    grid = (min(sms, g * triton.cdiv(t, 16) * triton.cdiv(r, 64)),)
    bmm_kernel_persistent[grid](
        x, weight, out, g, t, r, d,
        x.stride(1), x.stride(0), x.stride(2),
        weight.stride(0), weight.stride(2), weight.stride(1),
        out.stride(1), out.stride(0), out.stride(2),
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=64, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1, NUM_SMS=sms,
        A_LARGE=x.numel() > 2**31, B_LARGE=weight.numel() > 2**31,
        C_LARGE=out.numel() > 2**31, num_warps=4, num_stages=1,
    )
    return out
