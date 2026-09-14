"""Default-off TP8 large-prefill wo_a with a fixed K reduction order.

The BM128/BN128/BK128/8-wave tile is the validated offline candidate. This is
a numerical diagnostic, not a replacement for small-prefill/decode kernels.
"""
import logging

import torch
import triton
import triton.language as tl

_logged = False


@triton.jit(do_not_specialize=["M"])
def _project(X, W, Y, M):
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    k = tl.arange(0, 128)
    acc = tl.zeros((128, 128), tl.float32)
    for start in range(4096 // 128):
        kk = start * 128 + k
        a = tl.load(X + m[:, None] * 4096 + kk[None, :], m[:, None] < M, other=0)
        b = tl.load(W + n[None, :] * 4096 + kk[:, None])
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * 1024 + n[None, :], acc, m[:, None] < M)


def project(x, weight):
    global _logged
    if (x.ndim != 2 or weight.shape != (1024, 4096)
            or x.shape[1] != 4096 or not 8192 <= x.shape[0] <= 36864
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.device != weight.device or not x.is_cuda
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError("Stable wo_a requires BF16 M8192..36864/K4096/N1024 on one GPU")
    y = torch.empty((x.shape[0], 1024), device=x.device, dtype=x.dtype)
    _project[(triton.cdiv(x.shape[0], 128), 8)](
        x, weight, y, x.shape[0], num_warps=8, num_stages=2)
    if not _logged:
        logging.getLogger(__name__).info(
            "DSV4 diagnostic stable wo_a selected: M=%d K4096 N1024 tile128/128/128 waves8",
            x.shape[0])
        _logged = True
    return y
