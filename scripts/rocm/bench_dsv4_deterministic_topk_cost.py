"""Synthetic Top-K GPU cost: deterministic reference/candidate ABBA by default.

Optional legacy mode 0 is not a correctness reference. This is not E2E latency.
Many sequential nodes per graph amortize Python replay/submission overhead.
"""
import json
import os
import argparse

import torch
import sgl_kernel  # Register HIP AOT operations for the legacy arm.

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--modes', nargs=2, default=['2', '3'])
    parser.add_argument('--batches', nargs='+', type=int, default=[1, 32])
    parser.add_argument('--lengths', nargs='+', type=int, default=[576, 4096, 16384, 65536])
    args = parser.parse_args()
    torch.manual_seed(7)
    nodes, replays = 50, 20
    for batch, length in ((b, n) for b in args.batches for n in args.lengths):
        scores = torch.randn(batch, length, device="cuda")
        lens = torch.full((batch,), length, device="cuda", dtype=torch.int32)
        pages = torch.arange((length + 63) // 64, device="cuda", dtype=torch.int32)[None, :].expand(batch, -1).contiguous()
        out = torch.empty((batch, 512), device="cuda", dtype=torch.int32)
        measurements = []
        for mode in (args.modes[0], args.modes[1], args.modes[1], args.modes[0]):
            os.environ["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = mode
            for _ in range(5):
                topk_transform_512(scores, lens, pages, out, 64)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(nodes):
                    topk_transform_512(scores, lens, pages, out, 64)
            graph.replay()
            torch.cuda.synchronize()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            for _ in range(replays):
                graph.replay()
            end.record()
            end.synchronize()
            measurements.append(dict(mode=mode, us=begin.elapsed_time(end) * 1000 / (nodes * replays)))
        print(json.dumps(dict(logical_kv=length, batch=batch, ABBA=measurements)), flush=True)


if __name__ == "__main__":
    main()
