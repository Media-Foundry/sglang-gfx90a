"""Opt-in original-V4 prefill candidate; FP32 math, not legacy bit-exact."""
import logging
import os
from functools import lru_cache

import torch
import triton
import triton.language as tl

from sglang.kernels.jit.utils import load_jit

logger = logging.getLogger(__name__)
_logged = False


@triton.jit
def _finish(parts, rms, out, M, EPS: tl.constexpr):
    row = tl.program_id(0)
    c = tl.arange(0, 32)
    acc = tl.full((32,), 0, tl.float32)
    for s in tl.static_range(16):
        acc += tl.load(parts + (s * M + row) * 24 + c, c < 24, 0)
    p = tl.arange(0, 64)
    inv = tl.rsqrt(tl.sum(tl.load(rms + row * 64 + p), 0) / 16384 + EPS)
    tl.store(out + row * 24 + c, acc * inv, c < 24)


@lru_cache(maxsize=1)
def premix_mfma_module():
    return load_jit(
        "dsv4_mhc_premix_mfma16",
        cuda_files=["deepseek_v4/gfx90a_mhc_premix_mfma.cuh"],
        cuda_wrappers=[("run", "sglang::MhcCooperativeScreen::s16")],
        extra_cuda_cflags=["-O3", "-fno-fast-math", "-ffp-contract=off"],
    )


def try_premix_mfma(residual, fn, rms_partials, out, rms_eps):
    if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA", "0") != "1":
        return False
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active

    m = residual.shape[0]
    if not mix_pair_active() or not 8192 <= m <= 65536:
        return False
    if residual.data_ptr() % 16 or fn.data_ptr() % 16:
        return False
    # Called only after premix_reuse4's device/dtype/shape/contiguity contract.
    # Per-call ownership: no shared cross-layer/stream workspace or stale cache.
    scratch = torch.empty((16, m, 24), dtype=torch.float32, device=residual.device)
    premix_mfma_module().run(residual.view(m, 16384), fn, scratch)
    _finish[(m,)](scratch, rms_partials, out, m, float(rms_eps), num_warps=1)
    global _logged
    if not _logged:
        logger.info("DSV4 cooperative FP32 pre-mix selected: rows=%d split=16", m)
        _logged = True
    if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA_CHECK", "0") == "1":
        from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_pair import premix8_pair

        ref = torch.empty_like(out)
        premix8_pair[(12, triton.cdiv(m, 8))](
            residual, fn, rms_partials, ref, m, float(rms_eps), num_warps=1
        )
        delta = out - ref
        maximum = delta.abs().max().item()
        relative = (delta.norm() / ref.norm().clamp_min(1.e-30)).item()
        if not torch.isfinite(out).all().item() or relative > 1.e-5:
            raise RuntimeError(f"MHC MFMA live mix failed: max_abs={maximum} rel_l2={relative}")
        logger.info("DSV4 MFMA live mix: rows=%d max_abs=%.9g rel_l2=%.9g", m, maximum, relative)
    return True
