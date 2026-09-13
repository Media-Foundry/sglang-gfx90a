"""Opt-in fixed-tile BF16 router GEMM for V4.1's [384,5120] gate."""

import torch
import triton


def router_bf16_invariant(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (x.ndim != 2 or x.shape[1] != 5120 or weight.shape != (384,5120)
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.device != weight.device or x.device.type != 'cuda'
            or torch.version.hip is None or x.stride(1) != 1 or not weight.is_contiguous()):
        raise ValueError('Expected HIP BF16 router x[M,5120] and weight[384,5120]')
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import bmm_kernel_persistent

    m=x.shape[0]
    out=torch.empty((m,384),device=x.device,dtype=x.dtype)
    if not m:
        return out
    sms=torch.cuda.get_device_properties(x.device).multi_processor_count
    bmm_kernel_persistent[(min(sms,triton.cdiv(m,16)*6),)](
        x,weight,out,1,m,384,5120,
        0,x.stride(0),x.stride(1),0,weight.stride(1),weight.stride(0),
        0,out.stride(0),out.stride(1),
        BLOCK_SIZE_M=16,BLOCK_SIZE_N=64,BLOCK_SIZE_K=64,
        GROUP_SIZE_M=1,NUM_SMS=sms,
        A_LARGE=x.numel()>2**31,B_LARGE=False,C_LARGE=out.numel()>2**31,
        num_warps=4,num_stages=1,
    )
    return out
