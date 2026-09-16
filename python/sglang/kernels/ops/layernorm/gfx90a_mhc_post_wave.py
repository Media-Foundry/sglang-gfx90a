"""Opt-in original-V4 large-prefill post, matching the validated FP32/DPP tree."""
import logging
import os

import torch

from sglang.kernels.jit.utils import cache_once, load_jit


@cache_once
def post_wave_module():
    return load_jit(
        "gfx90a_mhc_post_wave",
        cuda_files=["deepseek_v4/gfx90a_mhc_post_wave.cuh"],
        cuda_wrappers=[("run", "sglang::Gfx90aMhcPostWave::run")],
        extra_cuda_cflags=["-O3", "-fno-fast-math", "-ffp-contract=off"],
    )


@cache_once
def _announce():
    logging.getLogger(__name__).warning("DSV4 exact HIP post-wave selected (native large prefill)")


def run_post_wave(x, residual, post, comb, out, partials):
    # The existing fused4 wrapper validates tensor layouts and gfx90a, and
    # requires the strict native-EXTEND context before calling this function.
    _announce()
    post_wave_module().run(x, residual, post, comb, out, partials)
    if os.getenv("SGLANG_DSV4_DEBUG_POST_WAVE_CHECK", "0") == "1":
        from .gfx90a_mhc_post_fused4 import _post_combine_fused4

        reference = torch.empty_like(out)
        reference_partials = torch.empty_like(partials)
        _post_combine_fused4[(x.shape[0], 16)](
            x, residual, post, comb, reference, reference_partials, num_warps=4
        )
        if not torch.equal(out.view(torch.uint8), reference.view(torch.uint8)):
            raise RuntimeError("HIP post-wave residual differs from fused4 reference")
        if not torch.equal(partials.view(torch.int32), reference_partials.view(torch.int32)):
            raise RuntimeError("HIP post-wave RMS partials differ from fused4 reference")
        logging.getLogger(__name__).warning("DSV4 post wave exact: rows=%d", x.shape[0])
