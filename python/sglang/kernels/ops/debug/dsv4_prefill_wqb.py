"""Default-off fixed-order TP8 large-prefill K1024/N4096 projections."""
import logging

import torch
import triton
import triton.language as tl

_logged = set()


@triton.jit(do_not_specialize=["M"])
def _project(X, W, Y, M):
    m = tl.program_id(0) * 128 + tl.arange(0, 128)
    n = tl.program_id(1) * 128 + tl.arange(0, 128)
    k = tl.arange(0, 128)
    acc = tl.zeros((128, 128), tl.float32)
    for start in range(1024 // 128):
        kk = start * 128 + k
        a = tl.load(X + m[:, None] * 1024 + kk[None, :], m[:, None] < M, other=0)
        b = tl.load(W + n[None, :] * 1024 + kk[:, None])
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * 4096 + n[None, :], acc, m[:, None] < M)


def project(x, weight, *, projection_name="wq_b"):
    if projection_name not in ("wq_b", "wo_b"):
        raise ValueError("Unsupported diagnostic projection name")
    if (x.ndim != 2 or weight.shape != (4096, 1024)
            or x.shape[1] != 1024 or not 8192 <= x.shape[0] <= 36864
            or x.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16
            or x.device != weight.device or not x.is_cuda
            or not x.is_contiguous() or not weight.is_contiguous()):
        raise ValueError(f"Stable {projection_name} requires BF16 M8192..36864/K1024/N4096 on one GPU")
    y = torch.empty((x.shape[0], 4096), device=x.device, dtype=x.dtype)
    _project[(triton.cdiv(x.shape[0], 128), 32)](
        x, weight, y, x.shape[0], num_warps=8, num_stages=2)
    if projection_name not in _logged:
        logging.getLogger(__name__).info(
            "DSV4 diagnostic stable %s selected: M=%d K1024 N4096 tile128/128/128 waves8",
            projection_name, x.shape[0])
        _logged.add(projection_name)
    return y
