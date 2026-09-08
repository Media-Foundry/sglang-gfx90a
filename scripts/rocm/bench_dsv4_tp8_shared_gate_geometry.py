#!/usr/bin/env python3
"""TP8 C1 shared gate/up N512 K4096 isolated fixed-unroll screen."""
import json
import statistics
import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args


def main():
    torch.manual_seed(20908)
    x = torch.randn(1, 4096, device='cuda', dtype=torch.bfloat16)
    weights = [torch.randn(512, 4096, device='cuda', dtype=torch.bfloat16) for _ in range(43)]
    a = torch.empty(1, 512, device='cuda', dtype=torch.bfloat16)
    b = torch.empty_like(a)
    def module(rows, waves):
        args = make_cpp_args(1, 512, 4096, rows, 2, waves)
        return load_jit('gfx90a_bf16_gemv', *args,
            cuda_files=['gemm/gfx90a_bf16_gemv.cuh'],
            cuda_wrappers=[('run', f'sglang::Gfx90aBf16GemvKernel<{args}>::run')],
            extra_cuda_cflags=['-O3'])
    baseline = module(2, 8)
    for rows, waves in ((2,8), (1,4), (1,8), (2,4), (1,16), (2,16)):
        candidate = module(rows, waves)
        exact = 0
        max_abs = 0.0
        for _ in range(100):
            x.normal_()
            baseline.run(x, weights[0], a)
            candidate.run(x, weights[0], b)
            exact += int(torch.equal(a, b))
            max_abs = max(max_abs, (a.float()-b.float()).abs().max().item())
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga):
            for w in weights: baseline.run(x, w, a)
        with torch.cuda.graph(gb):
            for w in weights: candidate.run(x, w, b)
        ga.replay(); gb.replay()
        graph_exact = torch.equal(a,b)
        witness = b.clone()
        for _ in range(1000): gb.replay()
        stable = torch.equal(b,witness)
        samples = {'A': [], 'B': []}
        for _ in range(5):
            for name, graph in (('A',ga),('B',gb),('B',gb),('A',ga)):
                for _ in range(10): graph.replay()
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(20): graph.replay()
                end.record();end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/20/43)
        print(json.dumps(dict(rows=rows,waves=waves,unroll=2,exact=exact,
            max_abs=max_abs,graph_exact=graph_exact,stable_after_1000=stable,
            median_us={k:statistics.median(v) for k,v in samples.items()},
            samples_us=samples)),flush=True)


if __name__ == '__main__': main()
