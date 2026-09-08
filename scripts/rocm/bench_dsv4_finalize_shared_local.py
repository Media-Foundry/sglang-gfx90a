#!/usr/bin/env python3
"""Isolated M32 local reduction/shared-add oracle; not an E2E/AR measurement."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import _jit_down_grouped


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(2090801)
    partial = torch.randn(32, 6, 4096, device='cuda')
    shared = torch.randn(32, 4096, device='cuda', dtype=torch.bfloat16)
    a = torch.empty_like(shared)
    b = torch.empty_like(shared)
    baseline = _jit_down_grouped(256, 32, 6, 4096, 256, 4, 2, 8, 832, 2)
    cpp = make_cpp_args(32)
    candidate = load_jit('gfx90a_finalize_shared_local_oracle', cpp,
        cuda_files=['deepseek_v4/gfx90a_finalize_shared_oracle.cuh'],
        cuda_wrappers=[('run', f'sglang::Gfx90aFinalizeSharedOracle<{cpp}>::run')],
        extra_cuda_cflags=['-O3'])

    def ref():
        baseline.reduce(partial, a)
        a.add_(shared)

    def new():
        candidate.run(partial, shared, b)

    graphs = {}
    for name, fn in [('A', ref), ('B', new)]:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fn()
        graphs[name] = graph
    exact = 0
    max_abs = 0.0
    for i in range(100):
        partial.normal_().mul_(2.0 ** (i % 17 - 8))
        shared.normal_()
        if i % 10 == 0:
            # Cancellation stresses the exact subgroup sum and BF16 round boundary.
            partial[:, 4].copy_(-partial[:, 0])
            partial[:, 5].copy_(-partial[:, 1])
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
        exact += int(torch.equal(a.view(torch.int16), b.view(torch.int16)))
        max_abs = max(max_abs, float((a.float() - b.float()).abs().max()))
    assert exact == 100, (exact, max_abs)
    reference = b.clone()
    allocated = torch.cuda.memory_allocated()
    for _ in range(1000):
        graphs['B'].replay()
    torch.cuda.synchronize()
    assert torch.equal(reference.view(torch.int16), b.view(torch.int16))
    assert torch.cuda.memory_allocated() == allocated
    # Time a GPU-resident burst, not 100 Python submissions of a ~6-us graph.
    # Otherwise host launch gaps can hide a one-kernel saving.
    bursts = {}
    for name, fn in [('A', ref), ('B', new)]:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(100):
                fn()
        bursts[name] = graph
    samples = {'A': [], 'B': []}
    for name in ['A', 'B', 'B', 'A'] * 5:
        graph = bursts[name]
        for _ in range(10):
            graph.replay()
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        samples[name].append(start.elapsed_time(end) * 10)
    result = dict(mutations_exact=exact, max_abs=max_abs, replay1000_exact=True,
        allocation_stable=True, samples_us=samples,
        trimmed_us={k: statistics.mean(sorted(v)[1:-1]) for k, v in samples.items()},
        timing='100 operations captured per graph; five ABBA cycles',
        scope='M32 local only; shared already ready; no collective or service overlap')
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
