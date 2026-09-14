"""Continue an existing FP32 Sinkhorn8 comb to20 without reinitialization."""

import torch
import triton
import triton.language as tl


@triton.jit
def _refine12(comb, eps: tl.constexpr):
    row = tl.program_id(0)
    offsets = tl.arange(0, 16)
    matrix = tl.reshape(tl.load(comb + row * 16 + offsets), (4, 4))
    for _ in tl.static_range(12):
        matrix = matrix / (tl.sum(matrix, axis=1)[:, None] + eps)
        matrix = matrix / (tl.sum(matrix, axis=0)[None, :] + eps)
    tl.store(comb + row * 16 + offsets, tl.reshape(matrix, (16,)))


def refine_comb20(comb: torch.Tensor, eps: float) -> None:
    assert comb.ndim == 3 and comb.shape[1:] == (4, 4)
    assert comb.dtype == torch.float32 and comb.is_cuda and comb.is_contiguous()
    _refine12[(comb.shape[0],)](comb, eps=eps, num_warps=1)
