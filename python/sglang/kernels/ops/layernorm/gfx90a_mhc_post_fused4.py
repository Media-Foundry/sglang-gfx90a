"""Four-output H256 post-combine reuse, preserving the reference RMS partials."""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _post_combine_fused4(x, residual, post, comb, out, partials):
    token = tl.program_id(0)
    block = tl.program_id(1)
    h = block * 256 + tl.arange(0, 256)
    xv = tl.load(x + token * 4096 + h).to(tl.float32)
    base = residual + token * 16384 + h
    r0 = tl.load(base).to(tl.float32)
    r1 = tl.load(base + 4096).to(tl.float32)
    r2 = tl.load(base + 8192).to(tl.float32)
    r3 = tl.load(base + 12288).to(tl.float32)
    for hc in tl.static_range(4):
        pv = tl.load(post + token * 4 + hc)
        # Production comb axes are [input_channel, output_channel].
        cb = comb + token * 16 + hc
        c0 = tl.load(cb)
        c1 = tl.load(cb + 4)
        c2 = tl.load(cb + 8)
        c3 = tl.load(cb + 12)
        acc = pv * xv
        acc += c0 * r0 + c1 * r1 + c2 * r2 + c3 * r3
        tl.store(out + token * 16384 + hc * 4096 + h, acc)
        rounded = acc.to(tl.bfloat16).to(tl.float32)
        sq = tl.sum(rounded * rounded, 0)
        tl.store(partials + (token * 4 + hc) * 16 + block, sq)


def post_combine_fused4(x, residual, post, comb):
    m = x.shape[0]
    tensors = (x, residual, post, comb)
    if not (
        m > 0
        and x.shape == (m, 4096)
        and residual.shape == (m, 4, 4096)
        and post.shape == (m, 4)
        and comb.shape == (m, 4, 4)
        and x.dtype == residual.dtype == torch.bfloat16
        and post.dtype == comb.dtype == torch.float32
        and x.device.type == "cuda"
        and all(t.device == x.device and t.is_contiguous() for t in tensors)
        and bool(torch.version.hip)
        and "gfx90a" in torch.cuda.get_device_properties(x.device).gcnArchName
    ):
        return None
    out = torch.empty_like(residual)
    partials = torch.empty((m, 64), dtype=torch.float32, device=x.device)
    if os.getenv("SGLANG_DSV4_DEBUG_PREFILL_POST_WAVE", "0") == "1":
        from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active

        # Stronger than a shape check: original V4 / TP8 / native EXTEND only.
        if mix_pair_active() and 8192 <= m <= 65536:
            from .gfx90a_mhc_post_wave import run_post_wave

            run_post_wave(x, residual, post, comb, out, partials)
            return out, partials
    _post_combine_fused4[(m, 16)](
        x, residual, post, comb, out, partials, num_warps=4
    )
    return out, partials
