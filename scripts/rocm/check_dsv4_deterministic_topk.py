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
    parser.add_argument("--mode", choices=("2", "3"), default="2")
    parser.add_argument("--mutations", type=int, default=100)
    args = parser.parse_args()
    if args.default_mode:
        os.environ.pop("SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER", None)
    else:
        os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = args.mode
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
    # Replay both implementations with identical, mutated static buffers.
    # Exercise cutoff membership, changing live lengths and physical mappings.
    mode = os.environ.get("SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER")
    x = torch.empty((32, 4096), device="cuda")
    lengths = torch.full((32,), 4096, device="cuda", dtype=torch.int32)
    table = torch.arange(64, device="cuda", dtype=torch.int32).expand(32, -1).contiguous()
    outputs, graphs = [], []
    x.normal_()
    for candidate in ("2", "3"):
        os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = candidate
        physical = torch.empty((32, 512), device="cuda", dtype=torch.int32)
        logical = torch.empty_like(physical)
        topk_transform_512(x, lengths, table, physical, 64, logical)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_transform_512(x, lengths, table, physical, 64, logical)
        outputs.append((physical, logical))
        graphs.append(graph)
    for iteration in range(args.mutations):
        if iteration % 2:
            x.copy_(torch.randint(-4, 5, x.shape, device="cuda"))
        else:
            x.normal_()
        # Include signed zero and nonfinite keys in the equivalence oracle.
        x[:, :4] = torch.tensor([0., -0., float('inf'), float('nan')], device="cuda")
        lengths.random_(0, 4097)
        table.copy_(torch.randperm(64, device="cuda", dtype=torch.int32)[None, :])
        for graph, result in zip(graphs, outputs):
            for tensor in result:
                tensor.fill_(-777)
            graph.replay()
        assert all(torch.equal(a, b) for a, b in zip(*outputs)), iteration
    if mode is None:
        os.environ.pop("SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER", None)
    else:
        os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = mode
    print(f"mode 2/3 static graph input/length/page mutations: {args.mutations} PASS", flush=True)
    for width in (576, 4096):
        batch_scores = torch.randn(2304, width, device="cuda")
        batch_lens = torch.randint(0, width + 1, (2304,), device="cuda", dtype=torch.int32)
        batch_pages = torch.arange((width + 63) // 64, device="cuda", dtype=torch.int32)[None, :].expand(2304, -1).contiguous()
        check(batch_scores, batch_lens, batch_pages, min(args.replays, 10))
    if args.capture:
        d = torch.load(args.capture, map_location="cpu", weights_only=True)
        check(*(d[k].cuda() for k in ("logits", "seq_lens", "page_table")), args.replays)


if __name__ == "__main__":
    main()
