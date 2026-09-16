"""Opt-in common FP32/20 MHC arithmetic for scoped large native prefill.

The returned None is the existing no-singleton-optimization dispatch hint.
It does not change ForwardBatch, scheduling, communication or model weights.
It is not a claim of arbitrary batch-invariant floating-point computation.
"""
import logging
import os

from sglang.srt.environ import envs
from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active

logger = logging.getLogger(__name__)
_logged = set()
ENV = "SGLANG_DSV4_PREFILL_MHC_COMMON_FP32"


def common_prefill_batch_hint(batch_size, rows):
    # The scope enforces original V4, TP8/EP1, native eager EXTEND, no CP,
    # TBO, graph capture or draft. Do not infer those conditions from M alone.
    if os.getenv(ENV, "0") != "1" or not mix_pair_active() or not 8192 <= rows <= 65536:
        return batch_size
    conflicts = [name for name, active in (
        ("BF16 MHC dot", envs.SGLANG_DSV4_GFX90A_BF16_MHC_DOT.get()),
        ("MFMA pre-mix", os.getenv("SGLANG_DSV4_DEBUG_PREFILL_MIX_MFMA", "0") == "1"),
        ("custom Sinkhorn iterations", os.getenv("SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS", "0") == "1"),
        ("comb refinement", os.getenv("SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20", "0") == "1"),
    ) if active]
    if conflicts:
        raise ValueError("Common FP32/20 prefill conflicts with " + ", ".join(conflicts))
    if batch_size not in _logged:
        logger.info("DSV4 common FP32/20 prefill MHC selected: rows=%d scheduler_batch=%s dispatch_hint=None",
                    rows, batch_size)
        _logged.add(batch_size)
    # This consistently bypasses all legacy singleton paths (including full
    # native, fused-tail, split-K, native finish and fused weighted-RMS), and
    # makes hc_split_sinkhorn use the configured model's 20-iteration branch.
    return None
