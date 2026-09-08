#!/usr/bin/env python3
"""Isolated direct-X portability screen; never widens the production selector."""
import argparse
import json
import statistics
from pathlib import Path

import torch
from sglang.kernels.jit.utils import load_jit, make_cpp_args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(20908)
    results = {}
    for m in (1, 32):
        x = torch.randn(m, 1, 4096, device='cuda', dtype=torch.bfloat16)
        weights = [torch.randn(1, 1024, 4096, device='cuda', dtype=torch.bfloat16)
                   for _ in range(43)]
        modules = {}
        for name, stage in [('staged', True), ('direct', False)]:
            args = make_cpp_args(m, 1, 1024, 4096, 1, 2, 4, stage)
            modules[name] = load_jit(
                'gfx90a_woa_direct_oracle', *args,
                cuda_files=['gemm/gfx90a_bf16_grouped_direct_x_oracle.cuh'],
                cuda_wrappers=[('run', f'sglang::Gfx90aBf16GroupedGemvKernel<{args}>::run')],
                extra_cuda_cflags=['-O3'])
        outputs = {n: torch.empty(m, 1, 1024, device='cuda', dtype=torch.bfloat16)
                   for n in modules}
        def run(name, w):
            if name == 'einsum':
                return torch.einsum('tgd,grd->tgr', x, w)
            modules[name].run(x, w, outputs[name])
            return outputs[name]
        names = ['einsum', 'staged', 'direct']
        graphs = {}
        for name in names:
            for _ in range(3): run(name, weights[0])
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): y = run(name, weights[0])
            graphs[name] = g, y
        checks = []
        for i in range(100):
            x.normal_()
            if i % 25 == 0: weights[0].normal_()
            for g, _ in graphs.values(): g.replay()
            torch.cuda.synchronize()
            direct = outputs['direct'].clone()
            exact = torch.equal(direct, outputs['staged'])
            for _ in range(10): graphs['direct'][0].replay()
            ref = torch.matmul(x.float(), weights[0][0].float().T)
            checks.append(dict(exact=exact, replay_exact=torch.equal(direct, outputs['direct']),
                               finite=bool(torch.isfinite(direct).all()),
                               einsum_exact=torch.equal(direct, graphs['einsum'][1]),
                               relative_l2=float((direct.float()-ref).norm()/ref.norm())))
        assert all(c['exact'] and c['replay_exact'] and c['finite'] for c in checks)
        timing = {}
        for mode in ('hot', '43_weights'):
            bursts = {}
            for name in names:
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    for j in range(43): run(name, weights[j if mode == '43_weights' else 0])
                bursts[name] = g
            samples = {n: [] for n in names}
            for candidate in ('staged', 'direct'):
                for name in ['einsum', candidate, candidate, 'einsum'] * 3:
                    bursts[name].replay()
                    start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    start.record()
                    for _ in range(10): bursts[name].replay()
                    end.record(); end.synchronize()
                    samples[name].append(start.elapsed_time(end)*1000/430)
            timing[mode] = dict(samples_us=samples, median_us={n: statistics.median(v) for n,v in samples.items()})
        results[str(m)] = dict(checks=checks, timing=timing)
        a.output.write_text(json.dumps(results, indent=2)+'\n')
        print(m, {mode:v['median_us'] for mode,v in timing.items()}, flush=True)


if __name__ == '__main__':
    main()
