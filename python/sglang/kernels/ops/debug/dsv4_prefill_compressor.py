"""Default-off layer2 core-compressor row-stability ablation (not a speed path)."""
import logging
import os

import torch
import triton
import triton.language as tl

_logged = False
FLAG = 'SGLANG_DSV4_DEBUG_PREFILL_CORE_COMPRESSOR_STABLE'


def enabled(compressor, x, batch):
    if os.getenv(FLAG, '0') != '1':
        return False
    from sglang.kernels.ops.debug.dsv4_prefill_attention_ar import enabled_for
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return bool(compressor._debug_original_v4 and compressor.layer_id == 2
                and not compressor.is_in_indexer and compressor.ratio == 4
                and compressor.head_dim == 512 and parallel.attn_cp_size == 1
                and enabled_for(batch, x.device, x.shape[0], parallel.attn_tp_size,
                                flag=FLAG))


@triton.jit(do_not_specialize=['M'])
def _project(X, W, Y, M):
    m = tl.program_id(0)*128 + tl.arange(0,128)
    n = tl.program_id(1)*128 + tl.arange(0,128)
    k = tl.arange(0,128)
    acc = tl.zeros((128,128),tl.float32)
    for start in range(4096//128):
        kk = start*128+k
        a = tl.load(X+m[:,None]*4096+kk[None,:], m[:,None]<M, other=0)
        b = tl.load(W+n[None,:]*4096+kk[:,None])
        acc = tl.dot(a,b,acc)
    # Preserve the runtime contract: BF16 result, then promoted to FP32.
    result = acc.to(tl.bfloat16).to(tl.float32)
    tl.store(Y+m[:,None]*2048+n[None,:],result,m[:,None]<M)


def project(x, weight):
    global _logged
    if (x.ndim != 2 or x.shape[1] != 4096 or weight.shape != (2048,4096)
            or not 8192 <= x.shape[0] <= 36864
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or not x.is_cuda or x.device != weight.device
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError('Core compressor diagnostic requires BF16 M8192..36864 K4096 N2048')
    y = torch.empty((x.shape[0],2048),device=x.device,dtype=torch.float32)
    _project[(triton.cdiv(x.shape[0],128),16)](
        x,weight,y,x.shape[0],num_warps=8,num_stages=2)
    if not _logged:
        logging.getLogger(__name__).info(
            'DSV4 diagnostic stable core compressor selected: layer2 M=%d K4096 N2048',x.shape[0])
        _logged = True
    return y
