"""Fixed K-order BF16 projection oracle. No weights/workspaces cached."""
import torch
import triton
import triton.language as tl


@triton.jit
def _linear(X, W, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr):
    m = tl.program_id(0) * 64 + tl.arange(0, 64)
    n = tl.program_id(1) * 64 + tl.arange(0, 64)
    k = tl.arange(0, 64)
    acc = tl.zeros((64, 64), tl.float32)
    for start in range(K // 64):
        kk = start * 64 + k
        a = tl.load(X + m[:, None] * K + kk[None, :], m[:, None] < M, other=0)
        b = tl.load(W + n[None, :] * K + kk[:, None], n[None, :] < N, other=0)
        acc = tl.dot(a, b, acc)
    tl.store(Y + m[:, None] * N + n[None, :], acc,
             (m[:, None] < M) & (n[None, :] < N))


def row_stable_linear(x, weight):
    assert x.ndim == weight.ndim == 2
    assert x.dtype == weight.dtype == torch.bfloat16
    assert x.is_contiguous() and weight.is_contiguous()
    m, k = x.shape
    n, wk = weight.shape
    assert k == wk and k % 64 == 0 and n % 64 == 0
    y = torch.empty((m, n), dtype=x.dtype, device=x.device)
    if m:
        _linear[(triton.cdiv(m, 64), triton.cdiv(n, 64))](
            x, weight, y, m, n, k, num_warps=4, num_stages=2)
    return y
