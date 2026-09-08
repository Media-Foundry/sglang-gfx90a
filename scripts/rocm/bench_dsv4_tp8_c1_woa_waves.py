#!/usr/bin/env python3
"""Exact G1 wo_a wave geometry screen; hot vs 64-MiB rotating weights."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    _jit_gfx90a_bf16_grouped_gemv_module,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(2090803)
    weights = [torch.randn(1, 1024, 4096, device='cuda', dtype=torch.bfloat16)
               for _ in range(8)]
    inputs = [torch.randn(1, 1, 4096, device='cuda', dtype=torch.bfloat16)
              for _ in weights]
    outputs = {w: [torch.empty(1, 1, 1024, device='cuda', dtype=torch.bfloat16)
                   for _ in weights] for w in (2, 4, 8)}
    modules = {4: _jit_gfx90a_bf16_grouped_gemv_module(1, 1)}
    for waves in (2, 8):
        args = make_cpp_args(1, 1, 1024, 4096, 1, 2, waves)
        modules[waves] = load_jit('gfx90a_bf16_grouped_gemv', *args,
            cuda_files=['gemm/gfx90a_bf16_gemv.cuh'],
            cuda_wrappers=[('run', f'sglang::Gfx90aBf16GroupedGemvKernel<{args}>::run')],
            extra_cuda_cflags=['-O3'])
    def call(waves, i):
        modules[waves].run(inputs[i], weights[i], outputs[waves][i])

    graphs = {}
    for waves in modules:
        for i in range(8):
            call(waves, i)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for i in range(8):
                call(waves, i)
        graphs[waves] = g
    exact = {w: 0 for w in (2, 8)}
    for mutation in range(100):
        for x in inputs:
            x.normal_()
        if mutation % 10 == 0:
            weights[(mutation // 10) % 8].normal_()
        for g in graphs.values():
            g.replay()
        torch.cuda.synchronize()
        for waves in exact:
            assert all(torch.isfinite(o).all() for o in outputs[waves])
            exact[waves] += int(all(torch.equal(x.view(torch.int16), y.view(torch.int16))
                               for x, y in zip(outputs[4], outputs[waves])))
    assert all(v == 100 for v in exact.values()), exact
    stable = {w: [o.clone() for o in outputs[w]] for w in modules}
    allocated = torch.cuda.memory_allocated()
    for _ in range(1000):
        for g in graphs.values():
            g.replay()
    torch.cuda.synchronize()
    for w in modules:
        assert all(torch.equal(x.view(torch.int16), y.view(torch.int16))
                   for x, y in zip(stable[w], outputs[w]))
    assert allocated == torch.cuda.memory_allocated()
    timing = {}
    for mode in ('hot', 'rotating8'):
        bursts = {}
        for waves in modules:
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for i in range(64):
                    call(waves, 0 if mode == 'hot' else i % 8)
            bursts[waves] = g
        samples = {w: [] for w in modules}
        for waves in [4, 2, 8, 8, 2, 4] * 5:
            g = bursts[waves]
            for _ in range(3):
                g.replay()
            start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
            start.record()
            g.replay()
            end.record()
            end.synchronize()
            samples[waves].append(start.elapsed_time(end) * 1000 / 64)
        timing[mode] = dict(samples_us=samples,
            trimmed_us={w: statistics.mean(sorted(v)[1:-1]) for w, v in samples.items()})
    result = dict(mutations_raw_bits_exact=exact, replay1000_exact=True,
        allocation_stable=True, transient_weight_bytes=sum(w.numel()*w.element_size() for w in weights),
        timing=timing, scope='C1 TP8 G1 wo_a only; synthetic weights; no E2E or selector change')
    a.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
