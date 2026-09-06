"""Check canonical logical Top-K ordering and unchanged selected membership."""
import argparse
import os

import torch
import sgl_kernel  # Register the HIP AOT operators.

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", help="Optional real indexer replay .pt")
    parser.add_argument("--replays", type=int, default=100)
    args = parser.parse_args()
    torch.manual_seed(123)
    if args.capture:
        d = torch.load(args.capture, map_location="cpu", weights_only=True)
        scores, lens, pages = [d[k].cuda() for k in ("logits", "seq_lens", "page_table")]
    else:
        scores = torch.randn(256, 576, device="cuda")
        lens = torch.arange(256, device="cuda", dtype=torch.int32) + 400
        lens.clamp_(max=576)
        pages = torch.arange(16, device="cuda", dtype=torch.int32).expand(256, 16).contiguous()
    out = torch.empty((scores.shape[0], 512), dtype=torch.int32, device="cuda")
    raw = torch.empty_like(out)
    os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = "0"
    topk_transform_512(scores, lens, pages, out, 64, raw)
    selected = raw.sort(-1).values.clone()
    os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = "1"
    reference = None
    for _ in range(args.replays):
        topk_transform_512(scores, lens, pages, out, 64, raw)
        if reference is None:
            reference = out.clone()
        assert torch.equal(reference, out), "ordered physical outputs changed"
        assert torch.equal(selected, raw.sort(-1).values), "selected logical set changed"
        assert (raw[:, :-1] >= raw[:, 1:]).all().item(), "logical order not canonical"
    print(f"{args.replays}/{args.replays} exact replays; membership preserved")


if __name__ == "__main__":
    main()
