"""Synthetic single-row Top-K GPU cost: legacy versus deterministic, ABBA.

Legacy selection is not a correctness reference. This is not E2E latency.
Many sequential nodes per graph amortize Python replay/submission overhead.
"""
import json
import os

import torch
import sgl_kernel  # Register HIP AOT operations for the legacy arm.

from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512


def main():
    torch.manual_seed(7)
    nodes, replays = 50, 20
    for length in (576, 4096, 16384):
        scores = torch.randn(1, length, device="cuda")
        lens = torch.tensor([length], device="cuda", dtype=torch.int32)
        pages = torch.arange((length + 63) // 64, device="cuda", dtype=torch.int32)[None, :]
        out = torch.empty((1, 512), device="cuda", dtype=torch.int32)
        measurements = []
        for mode in ("0", "2", "2", "0"):
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
        print(json.dumps(dict(logical_kv=length, batch=1, ABBA=measurements)), flush=True)


if __name__ == "__main__":
    main()
