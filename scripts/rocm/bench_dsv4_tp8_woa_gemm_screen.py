#!/usr/bin/env python3
"""TP8 G1 wo_a: bounded multi-weight graph ABBA, no serving modifications."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--service-pid', type=int, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--hipblaslt', action='store_true', help='Screen supported solutions, then validate fastest three')
    a = p.parse_args()
    assert os.getenv('HIP_VISIBLE_DEVICES') == '4'
    service = psutil.Process(a.service_pid)
    owned = {service.pid, *[c.pid for c in service.children(recursive=True)]}
    seen = set()
    def walk(x):
        if isinstance(x, dict):
            if 'pid' in x: seen.add(int(x['pid']))
            for v in x.values(): walk(v)
        elif isinstance(x, list):
            for v in x: walk(v)
    walk(json.loads(subprocess.check_output(['amd-smi', 'process', '--json'])))
    assert not seen-owned, ('external GPU owners', seen-owned)

    import torch
    import torch.nn.functional as F
    from bench_dsv4_inverse_rope_woa_lower_bound import load_runtime_woa_shard
    from aiter.tuned_gemm import tgemm, hipb_gemm

    assert torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] == 'gfx90a'
    torch.manual_seed(20908)
    source = Path('/tmp/dsv4_tp8_rowstable_router_all_20260908/layer_20_rank_0_attn_inverse_rope.pt')
    raw = torch.load(source, map_location='cpu', weights_only=True)
    # A real prefill-state slice, not claimed to be a diverse decode fixture.
    assert raw.ndim == 3 and raw.shape[1:] == (8,512) and raw.shape[0] >= 32
    x = raw[:32].contiguous().view(32,1,4096).cuda()
    seed_x = x.clone()
    layers = [0,6,12,18,20,24,36,42]
    weights = [load_runtime_woa_shard(argparse.Namespace(
        model_dir=Path('/home/pc/models/modelscope'), layer=layer, rank=0
    )).cuda() for layer in layers]
    funcs = {
        'einsum': lambda w: torch.einsum('tgd,grd->tgr', x, w),
        'linear': lambda w: F.linear(x[:,0], w[0]).unsqueeze(1),
        'mm': lambda w: torch.mm(x[:,0], w[0].t()).unsqueeze(1),
        'aiter_tgemm': lambda w: tgemm.mm(x[:,0], w[0], otype=x.dtype).unsqueeze(1),
    }
    def capture(fn):
        for w in weights: fn(w)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g): out = [fn(w) for w in weights]
        return g, out
    baseline, ref = capture(funcs['einsum'])
    def time_us(g):
        for _ in range(5): g.replay()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(50): g.replay()
        end.record(); end.synchronize()
        return start.elapsed_time(end)*1000/(50*len(weights))
    result = {'layers':layers, 'input_source':str(source),
              'torch_version':str(torch.__version__), 'hip_version':str(torch.version.hip),
              'fixture':'real prefill slice; 8 real TP8 checkpoint weight shards',
              'weight_bytes':sum(w.numel()*w.element_size() for w in weights), 'results':[]}
    if a.hipblaslt:
        from aiter import tuned_gemm
        from aiter.ops.gradlib import hipb_create_extension, hipb_findallsols
        if not tuned_gemm.extensions_created:
            hipb_create_extension()
            tuned_gemm.extensions_created = True
        solutions = hipb_findallsols(x[:,0], weights[0][0].t(), out_dtype=x.dtype)
        assert len(solutions) <= 4096, ('screen requires narrower bounds',len(solutions))
        screen = []
        for solution in solutions:
            fn = lambda w, solution=solution: hipb_gemm(x[:,0], w[0], solution).unsqueeze(1)
            graph,values = capture(fn)
            baseline.replay(); graph.replay(); torch.cuda.synchronize()
            assert all(torch.isfinite(v).all() for v in values), solution
            us = statistics.median(time_us(graph) for _ in range(3))
            screen.append({'solution':solution, 'us':us,
                           'initial_exact':all(torch.equal(v,r) for v,r in zip(values,ref))})
            del graph,values
        result['screen'] = sorted(screen,key=lambda r:r['us'])
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print('supported solutions',len(screen),'top',result['screen'][:5],flush=True)
        funcs = {f'hipblaslt_{row["solution"]}':
                 (lambda w, solution=row['solution']: hipb_gemm(x[:,0],w[0],solution).unsqueeze(1))
                 for row in result['screen'][:3]}
    for label,fn in funcs.items():
        if label == 'einsum': continue
        g,out = capture(fn)
        x.copy_(seed_x)
        exact = 0; max_abs = 0.; max_rel = 0.
        for iteration in range(100):
            if iteration:
                x.copy_(seed_x + torch.randn_like(x)*0.1)
            baseline.replay(); g.replay(); torch.cuda.synchronize()
            good = True
            for y,r in zip(out,ref):
                assert torch.isfinite(y).all()
                good &= torch.equal(y,r)
                d = (y.float()-r.float())
                max_abs = max(max_abs,float(d.abs().max()))
                max_rel = max(max_rel,float(d.norm()/r.float().norm().clamp_min(1e-12)))
            exact += int(good)
        saved = [y.clone() for y in out]
        for _ in range(1000): g.replay()
        torch.cuda.synchronize()
        stable = all(torch.equal(y,z) for y,z in zip(out,saved))
        assert stable
        x.copy_(seed_x)
        ta,tb = [],[]
        for _ in range(7):
            ta.append(time_us(baseline));tb.append(time_us(g))
            tb.append(time_us(g));ta.append(time_us(baseline))
        row = {'candidate':label,'exact_mutations':exact,'mutations':100,
               'max_abs':max_abs,'max_relative_l2':max_rel,'graph1000_stable':stable,
               'baseline_us':statistics.median(ta),'candidate_us':statistics.median(tb),
               'baseline_samples':ta,'candidate_samples':tb}
        result['results'].append(row)
        a.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps({k:v for k,v in row.items() if not k.endswith('samples')}),flush=True)


if __name__ == '__main__': main()
