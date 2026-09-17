"""Fixed-solution QKV: multi-layer numeric checks and paired graph ABBA.

Uses actual checkpoint projection weights; inputs are captured layer0 rows plus
perturbations, not claimed to be live late-layer activations. No service edits.
"""
import ast
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

root = Path(__file__).resolve().parent
repo = root.parents[2]
assert os.environ.get('HIP_VISIBLE_DEVICES') == '5'
target = root/'confirm.json'
assert not target.exists()
for arm in ('A1', 'B', 'A2'):
    stop = root.parent/'dsv4_first_mix_table_lengths_20260917/32k'/arm/f'P32k-first-table-{arm}.stop.json'
    assert not json.loads(stop.read_text())['remaining']
screen = json.loads((root/'oracle.json').read_text())
assert screen['status'] == 'complete'
assert hashlib.sha256((root/'oracle.py').read_bytes()).hexdigest() == screen['source_sha256']

import torch
from safetensors import safe_open
from sglang.kernels.ops.attention.dsv4.gfx90a_rocblas_linear import Linear

torch.set_num_threads(1)
torch.set_grad_enabled(False)
assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 50*1024**3
torch.backends.cuda.preferred_blas_library('cublaslt')
# Reuse the exact ABI wrapper and fixture loading from the frozen successful
# screen, without executing its enumeration/timing body again.
tree = ast.parse((root/'oracle.py').read_text())
cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'Fixed')
namespace = {'Linear': Linear, 'C': C, 'torch': torch}
exec(compile(ast.Module(body=[cls], type_ignores=[]), str(root/'oracle.py'), 'exec'), namespace)
fixed = namespace['Fixed']()
digest = lambda v: hashlib.sha256(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
trace = root.parent/'dsv4_prefill_tail_contract_20260917/trace-v2-mixed/trace'
meta = torch.load(trace/'layer-0-rank-0-metadata.pt', weights_only=False)['metadata']
record = json.loads((trace/'layer-0-rank-0-prepare_full_input.json').read_text())
path = trace/record['values_file']
assert hashlib.sha256(path.read_bytes()).hexdigest() == record['file_sha256']
raw = torch.load(path, weights_only=False)['samples']
seed = raw[[meta['sampled_rows'].index(i) for i in range(64)]].contiguous().view(torch.bfloat16).cuda()
assert digest(seed) == screen['input_sha256']
model = Path('/home/pc/models/modelscope')
index = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
solutions = [-606601783, -606601866, -606601816]
assert all(s in screen['common_solutions'] for s in solutions)
report = dict(status='running', scope=__doc__, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    oracle_sha256=hashlib.sha256((root/'oracle.json').read_bytes()).hexdigest(), numeric=[], timings=[], weights=[])


def save():
    target.write_text(json.dumps(report, indent=2)+'\n')


def weight_for(layer):
    tensors = []
    for name in ('wq_a', 'wkv'):
        key = f'layers.{layer}.attn.{name}'
        with safe_open(str(model/index[key+'.weight']), framework='pt', device='cpu') as handle:
            w = handle.get_tensor(key+'.weight')
        with safe_open(str(model/index[key+'.scale']), framework='pt', device='cpu') as handle:
            scale = handle.get_tensor(key+'.scale')
        # Exact E8M0 powers of two, then the existing BF16 materialization contract.
        assert w.dtype == torch.float8_e4m3fn and scale.dtype == torch.float8_e8m0fnu
        assert (scale.view(torch.uint8) != 255).all()
        factor = torch.exp2(scale.view(torch.uint8).float()-127).repeat_interleave(128,0).repeat_interleave(128,1)
        tensors.append((w.float()*factor).bfloat16())
    weight = torch.cat(tensors).contiguous().cuda()
    if layer == 0:
        assert digest(weight) == screen['weight_sha256'], 'Checkpoint expansion must match actual loaded layer0'
    report['weights'].append(dict(layer=layer, sha256=digest(weight)))
    return weight


def check(a, b):
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    d = (a.double()-b.double()).abs()
    return dict(exact=torch.equal(a,b), different=int((a != b).sum()), max_abs=float(d.max()))


def benchmark(rows, weight, solution):
    outputs = {name: torch.empty((len(rows),1536), device=rows.device, dtype=rows.dtype) for name in ('A','B')}
    calls = {'A': lambda: torch.mm(rows, weight.t(), out=outputs['A']),
             'B': lambda: fixed.run(rows, weight, solution, outputs['B'])}
    graphs = {}
    for name, call in calls.items():
        for _ in range(3): call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(5): call()
        graphs[name] = graph
    rounds = []
    for _ in range(4):
        values = {'A': [], 'B': []}
        for name in ('A','B','B','A'):
            start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record(); graphs[name].replay(); end.record(); end.synchronize()
            values[name].append(start.elapsed_time(end)/5)
        rounds.append(values)
    a = statistics.median(v for r in rounds for v in r['A'])
    b = statistics.median(v for r in rounds for v in r['B'])
    return dict(m=len(rows), solution=solution, control_ms=a, candidate_ms=b,
                time_reduction_pct=100*(1-b/a), rounds=rounds,
                versus_old=check(outputs['B'], outputs['A']))


save()
try:
    for layer in (0, 1, 10, 20, 30, 42):
        weight = weight_for(layer)
        for mutation in range(3):
            small = (seed.float()*(1-mutation*.011)).bfloat16().contiguous()
            double = small.double() @ weight.double().t()
            for solution in solutions:
                base = fixed.run(small,weight,solution)
                for m in (1, 64, 65, 8192, 32767):
                    rows = small.repeat((m+63)//64,1)[:m].contiguous()
                    result = fixed.run(rows,weight,solution)
                    perm = torch.randperm(m,device='cuda')
                    reordered = fixed.run(rows[perm].contiguous(),weight,solution)
                    entry = dict(layer=layer, mutation=mutation, solution=solution, m=m,
                        shape=check(result[:min(m,64)],base[:min(m,64)]),
                        row_order=check(reordered,result[perm]), fp64=check(result[:min(m,64)],double[:min(m,64)]))
                    report['numeric'].append(entry)
                save()
        if layer == 0:
            for m in (64,65,8192,32767,32768):
                rows = seed.repeat((m+63)//64,1)[:m].contiguous()
                for solution in solutions:
                    item = benchmark(rows,weight,solution)
                    report['timings'].append(item); save(); print('ABBA', item, flush=True)
        print('LAYER COMPLETE', layer, flush=True)
    report['all_shape_row_exact'] = all(v['shape']['exact'] and v['row_order']['exact'] for v in report['numeric'])
    report['status'] = 'complete'
    save()
finally:
    if report['status'] != 'complete':
        report['status'] = 'failed'; report['error'] = traceback.format_exc(); save()
    fixed.close()
