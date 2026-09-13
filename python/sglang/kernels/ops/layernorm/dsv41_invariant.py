"""Fixed per-row HC reductions for opt-in V4.1 HIP diagnostics."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rstd_kernel(X, Y, K: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    rstd = tl.rsqrt(tl.sum(x * x, 0) / K + EPS)
    tl.store(Y + row, rstd)


def hc_rstd_invariant(x: torch.Tensor, eps: float) -> torch.Tensor:
    if (x.ndim != 2 or x.shape[1] != 20480 or not x.is_contiguous()
            or x.dtype not in (torch.bfloat16, torch.float32)
            or x.device.type != 'cuda' or torch.version.hip is None):
        raise ValueError('Expected contiguous HIP V4.1 HC rows [M,20480], BF16/FP32')
    out = torch.empty((x.shape[0], 1), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        _rstd_kernel[(x.shape[0],)](x, out, x.shape[1], eps, 32768,
                                  num_warps=8, enable_fp_fusion=False)
    return out


def hc_mix_stats_invariant(x: torch.Tensor, fn: torch.Tensor, eps: float) -> torch.Tensor:
    if fn.shape != (24, 20480) or fn.device != x.device:
        raise ValueError('Expected resident HC weights [24,20480] on the activation device')
    from sglang.kernels.ops.layernorm.mhc import _gfx90a_mhc_mix_kernel

    # Fixing only RMS is sufficient for the captured M1/M204 dot, but the
    # mixed-row M64 oracle also exposes a GEMM row-placement difference.
    # Reuse the fixed FP32 dot reduction, not a shape-selected BLAS tactic.
    rstd = hc_rstd_invariant(x, eps)
    fn = fn.float().contiguous()
    out = torch.empty((x.shape[0], 24), device=x.device, dtype=torch.float32)
    if x.shape[0]:
        _gfx90a_mhc_mix_kernel[(6, x.shape[0])](
            x, fn, rstd, out, 20480, 24, BLOCK_N=4, BLOCK_K=256,
            num_warps=4, enable_fp_fusion=False,
        )
    return out
