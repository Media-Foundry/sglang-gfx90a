#!/usr/bin/env python3
"""Isolated AIter sort buffer-elision screen; never changes production."""
import json
import statistics

import torch
from aiter.fused_moe import moe_sorting


def main():
    torch.manual_seed(20260908)
    for m in (1, 32):
        ids = torch.empty((m, 6), dtype=torch.int32, device="cuda")
        weights = torch.empty((m, 6), dtype=torch.float32, device="cuda")
        ids.copy_(torch.rand((m, 256), device="cuda").argsort(dim=1)[:, :6])
        weights.uniform_()
        graphs, results = [], []
        for dim in (4096, 0):
            moe_sorting(ids, weights, 256, dim, torch.bfloat16, block_size=4)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = moe_sorting(ids, weights, 256, dim, torch.bfloat16, block_size=4)
            graphs.append(graph)
            results.append(result)
        for iteration in range(100):
            ids.copy_(torch.rand((m, 256), device="cuda").argsort(dim=1)[:, :6])
            weights.uniform_()
            for graph in graphs:
                graph.replay()
            torch.cuda.synchronize()
            a, b = results
            assert torch.equal(a[3], b[3]), (m, iteration, "valid")
            valid = int(a[3][0])
            assert 0 < valid <= a[0].numel() and valid % 4 == 0
            for field, length in ((0, valid), (1, valid), (2, valid // 4)):
                assert torch.equal(a[field][:length], b[field][:length]), (m, iteration, field)
        expected = [x.clone() for x in results[1][:4]]
        for _ in range(1000):
            graphs[1].replay()
        torch.cuda.synchronize()
        for field, length in ((0, valid), (1, valid), (2, valid // 4), (3, 2)):
            assert torch.equal(expected[field][:length], results[1][field][:length])
        samples = [[], []]
        for _ in range(5):
            for arm in (0, 1, 1, 0):
                for _ in range(20):
                    graphs[arm].replay()
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(100):
                    graphs[arm].replay()
                end.record()
                end.synchronize()
                samples[arm].append(start.elapsed_time(end) * 10)
        print(json.dumps(dict(m=m, exact_mutations=100, stable_replays=1000,
                              samples_us=samples,
                              trimmed_us=[statistics.mean(sorted(s)[1:-1]) for s in samples],
                              unused_buffer_bytes=m*4096*2)), flush=True)


if __name__ == "__main__":
    main()
