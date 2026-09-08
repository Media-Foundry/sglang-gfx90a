#!/usr/bin/env python3
"""Standalone fused C1 split-count screen; numerical differences are reported."""
import json
import statistics

import torch
from bench_dsv4_tp4_m32_paged_decode_geometry import capture, time_graph
from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.paged_decode import (
    _sparse_attn_v4_paged_decode_triton,
    _kv_splits_heuristic,
)


def main():
    torch.manual_seed(20260908)
    assert _kv_splits_heuristic(1, 8, 16) == 64
    for length in (128, 256, 640):
        q = torch.randn((1, 8, 512), device="cuda", dtype=torch.bfloat16)
        kv = torch.randn((length, 512), device="cuda", dtype=torch.bfloat16)
        ids = torch.arange(length, device="cuda", dtype=torch.int32)
        ptr = torch.tensor([0, length], device="cuda", dtype=torch.int32)
        sink = torch.randn(8, device="cuda")
        angles = torch.randn((1024, 32), device="cuda")
        freqs = torch.stack((angles.cos(), angles.sin()), -1).flatten(1).contiguous()
        positions = torch.tensor([length], device="cuda", dtype=torch.int64)
        graphs, outputs = {}, {}
        for split in (64, 32, 16):
            def run(split=split):
                return _sparse_attn_v4_paged_decode_triton(
                    q, kv, ids, ptr, sink, 512**-0.5, block_h=16,
                    block_k=16, kv_splits=split, _oracle_num_warps=4,
                    _oracle_num_stages=2, inverse_rope_freqs=freqs,
                    inverse_rope_positions=positions)
            graphs[split], outputs[split] = capture(run)
        samples = {s: [] for s in graphs}
        for _ in range(5):
            for split in (64, 32, 16, 16, 32, 64):
                samples[split].append(time_graph(graphs[split], 20, 100))
        exact = {s: 0 for s in graphs}
        max_abs = {s: 0.0 for s in graphs}
        max_rel_l2 = {s: 0.0 for s in graphs}
        for n in range(100):
            q.normal_(); kv.normal_(); sink.normal_()
            ids.random_(0, length)
            ptr[1] = 0 if n % 10 == 0 else (n * 37) % (length + 1)
            positions.random_(0, 1024)
            for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize()
            ref = outputs[64].float()
            for split, out in outputs.items():
                assert torch.isfinite(out).all(), (length, n, split)
                exact[split] += int(torch.equal(outputs[64], out))
                diff = out.float() - ref
                max_abs[split] = max(max_abs[split], diff.abs().max().item())
                max_rel_l2[split] = max(max_rel_l2[split],
                    (diff.norm() / ref.norm().clamp_min(1e-10)).item())
        for split, graph in graphs.items():
            expected = outputs[split].clone()
            for _ in range(1000): graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(expected, outputs[split])
        print(json.dumps(dict(context=length, samples_us=samples,
            trimmed_us={s: statistics.mean(sorted(v)[1:-1]) for s,v in samples.items()},
            exact_mutations=exact, max_abs=max_abs, max_relative_l2=max_rel_l2,
            stable_replays=1000, mutation_count=100)), flush=True)


if __name__ == "__main__":
    main()
