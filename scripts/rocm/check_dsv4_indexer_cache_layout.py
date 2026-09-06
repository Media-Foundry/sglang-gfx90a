"""Valid indexer logits must not depend on unwritten bytes in a partial page."""
import argparse

import torch

from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
from sglang.srt.layers.attention.dsa.utils import aiter_can_use_preshuffle_paged_mqa
from sglang.srt.layers.attention.dsv4.indexer import (
    FP8_DTYPE, _fp8_paged_mqa_logits_triton, fp8_paged_mqa_logits_torch,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, default=1)
    parser.add_argument("--fallback", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(123)
    m, h, length = 16, 64, args.tokens
    num_pages = (length + 63) // 64
    x = torch.randn(m, length, 128, device="cuda", dtype=torch.float32)
    q = torch.randn(m, 1, h, 128, device="cuda").to(FP8_DTYPE)
    w = torch.randn(m, h, device="cuda")
    loc = (torch.arange(m, device="cuda", dtype=torch.int32)[:, None] * (num_pages * 64)
           + torch.arange(length, device="cuda", dtype=torch.int32)[None, :]).flatten()
    pages = torch.arange(m * num_pages, device="cuda", dtype=torch.int32).reshape(m, num_pages)
    lens = torch.full((m,), length, device="cuda", dtype=torch.int32)
    results = []
    for fill in (0, 1):
        cache = torch.zeros(m * num_pages, 64 * 132, device="cuda", dtype=torch.uint8)
        if fill:
            cache.random_(0, 128)
        triton_fused_store_indexer(x.flatten(0, 1), cache, loc, 64)
        packed = cache.view(m * num_pages, 64, 1, 132)
        if args.fallback:
            result = fp8_paged_mqa_logits_torch(q, packed, w, lens, pages, None, length, False)
        else:
            result = _fp8_paged_mqa_logits_triton(q, packed, w, lens, pages, length)
        assert result is not None, "Full Triton selector must be enabled"
        results.append(result)
    a, b = results
    torch.cuda.synchronize()
    print(dict(preshuffle=aiter_can_use_preshuffle_paged_mqa(), tokens=length,
               exact=torch.equal(a, b), max_abs=(a-b).abs().max().item()), flush=True)
    assert torch.equal(a, b), "Indexer read unwritten cache bytes"
    if not args.fallback:
        # Reference the actual stored quantized values: x/scale versus the
        # writer's x*reciprocal(scale) can round differently at FP8 midpoints.
        # This test covers the reader/layout, not the quantizer's rounding rule.
        values = cache[:, :8192].contiguous().view(FP8_DTYPE)
        if aiter_can_use_preshuffle_paged_mqa():
            values = values.reshape(m, num_pages, 4, 8, 16, 16).permute(0, 1, 2, 4, 3, 5).contiguous()
        quantized = values.reshape(m, num_pages * 64, 128)[:, :length].float()
        scale = cache[:, 8192:].contiguous().view(torch.float32).reshape(m, num_pages * 64)[:, :length]
        scores = torch.einsum("btd,bhd->bth", quantized, q[:, 0].float())
        expected = (scores.clamp_min(0) * w[:, None, :]).sum(-1) * scale
        torch.testing.assert_close(a, expected, rtol=1e-4, atol=1e-4)
        print("FP32 reference passed", flush=True)


if __name__ == "__main__":
    main()
