"""HIP Top-K versus stable score/ID ordering, including cutoff ties and graphs."""
import argparse
import json
import os
from pathlib import Path

import torch

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512


def check(scores, lens, pages, replays):
    out = torch.empty((scores.shape[0], 512), device="cuda", dtype=torch.int32)
    raw = torch.empty_like(out)
    expected = torch.full_like(raw, -1)
    # CPU oracle deliberately independent of the GPU radix implementation.
    for row, length in enumerate(lens.cpu().tolist()):
        length = max(length, 0)
        values = scores[row, :length].cpu()
        ids = torch.argsort(values, descending=True, stable=True)[:512]
        ids = ids.sort(descending=True).values
        expected[row, :ids.numel()] = ids.cuda().int()
    def run():
        topk_transform_512(scores, lens, pages, out, 64, raw)
    run()
    assert torch.equal(raw, expected), (raw != expected).nonzero()[:20]
    slots = pages.gather(1, expected.clamp_min(0).long() // 64) * 64 + expected % 64
    slots.masked_fill_(expected < 0, -1)
    assert torch.equal(slots, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for _ in range(replays):
        raw.fill_(-777)
        out.fill_(-777)
        graph.replay()
        assert torch.equal(raw, expected)
        assert torch.equal(out, slots)
    print(f"shape={tuple(scores.shape)}: stable reference + {replays} graph replays PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture")
    parser.add_argument("--replays", type=int, default=100)
    parser.add_argument("--default-mode", action="store_true", help="Test gfx90a auto-selection without an override")
    args = parser.parse_args()
    if args.default_mode:
        os.environ.pop("SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER", None)
    else:
        os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = "2"
    torch.manual_seed(42)
    lens = torch.tensor([0, 1, 127, 512, 513, 577, 1024, 2049], device="cuda", dtype=torch.int32)
    pages = torch.randperm(80, device="cuda", dtype=torch.int32).expand(8, -1).contiguous()[:, :64]
    scores = torch.randn(8, 2080, device="cuda")[:, :2049]
    for mode in ("random", "all_ties", "quantized"):
        if mode == "all_ties":
            scores.fill_(-0.25)
        elif mode == "quantized":
            scores.copy_(torch.randint(-4, 5, scores.shape, device="cuda"))
        check(scores, lens, pages, args.replays)
    # Real capacity can be much larger than the live length. Invalid scores
    # must not enter selection, and completely tied rows must not overflow a
    # fixed-size shared-memory candidate queue.
    long_scores = torch.zeros(3, 65536, device="cuda")
    long_lens = torch.tensor([513, 32768, 65536], device="cuda", dtype=torch.int32)
    long_pages = torch.randperm(1024, device="cuda", dtype=torch.int32).expand(3, -1).contiguous()
    check(long_scores, long_lens, long_pages, args.replays)
    fixture = json.loads((Path(__file__).parent / "fixtures/dsv4_topk_layer8_row2144.json").read_text())
    real_scores = torch.tensor([fixture["scores"]], device="cuda", dtype=torch.float32)
    real_lens = torch.tensor([fixture["length"]], device="cuda", dtype=torch.int32)
    real_pages = torch.randperm((fixture["length"] + 63) // 64, device="cuda", dtype=torch.int32)[None, :]
    check(real_scores, real_lens, real_pages, args.replays)
    if args.capture:
        d = torch.load(args.capture, map_location="cpu", weights_only=True)
        check(*(d[k].cuda() for k in ("logits", "seq_lens", "page_table")), args.replays)


if __name__ == "__main__":
    main()
