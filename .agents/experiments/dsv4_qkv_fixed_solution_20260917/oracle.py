"""Bounded rocBLAS fixed-solution QKV screen; isolated, no service changes.

ABI checked against installed rocblas-beta.h:147 and rocblas-functions.h:21766.
Index semantics checked at the installed build commit 8d1ae90e: positive means
hipBLASLt, <-10 means explicit Tensile, 0/-1 means default. Public header prose
about all negative IDs is stale for this build.
No global BLAS preference change outside this isolated process. Fixed solution
is not assumed invariant: test shape, row permutation, replay, and FP64 error.
"""
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics
import traceback

assert os.environ.get('HIP_VISIBLE_DEVICES') == '5'
root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root/'oracle.json'
assert not target.exists()
assert '8d1ae90e' in Path('/opt/rocm/include/rocblas/internal/rocblas-version.h').read_text()
for arm in ('A1', 'B', 'A2'):
    stopped = root.parent/'dsv4_first_mix_table_lengths_20260917/16k'/arm/f'P16k-first-table-{arm}.stop.json'
    assert not json.loads(stopped.read_text())['remaining']

import torch
from sglang.kernels.ops.attention.dsv4.gfx90a_rocblas_linear import Linear

torch.set_num_threads(1)
torch.set_grad_enabled(False)
assert torch.cuda.device_count() == 1 and torch.cuda.mem_get_info()[0] > 50*1024**3
torch.backends.cuda.preferred_blas_library('cublaslt')


class Fixed(Linear):
    def __init__(self):
        super().__init__()
        # First 22 args are the gemm_ex contract through algo; replace index/flags
        # with flags + output solution list + in/out list size.
        self.lib.rocblas_gemm_ex_get_solutions.argtypes = self.lib.rocblas_gemm_ex.argtypes[:22] + [
            C.c_uint32, C.POINTER(C.c_int), C.POINTER(C.c_int)]

    def arguments(self, x, w, out):
        assert x.is_contiguous() and w.is_contiguous() and out.is_contiguous()
        assert x.dtype == w.dtype == out.dtype == torch.bfloat16
        assert x.device == w.device == out.device
        m, k = x.shape
        n = w.shape[0]
        assert w.shape[1] == k and out.shape == (m, n)
        self._check(self.lib.rocblas_set_stream(self.handle, C.c_void_p(torch.cuda.current_stream().cuda_stream)))
        return [self.handle, 112, 111, n, m, k,
                C.byref(self.alpha), C.c_void_p(w.data_ptr()), 168, k,
                C.c_void_p(x.data_ptr()), 168, k,
                C.byref(self.beta), C.c_void_p(out.data_ptr()), 168, n,
                C.c_void_p(out.data_ptr()), 168, n, 151, 1]

    def solutions(self, x, w):
        out = torch.empty((len(x), len(w)), device=x.device, dtype=x.dtype)
        args = self.arguments(x, w, out)
        size = C.c_int()
        query = self.lib.rocblas_gemm_ex_get_solutions
        self._check(query(*args, 0, None, C.byref(size)))
        assert 0 <= size.value <= 100000
        values = (C.c_int * size.value)()
        self._check(query(*args, 0, values, C.byref(size)))
        return sorted(set(values[:size.value]))

    def run(self, x, w, solution, out=None):
        assert solution > 0 or solution < -10, 'Default/reserved index is not a fixed solution'
        if out is None:
            out = torch.empty((len(x), len(w)), device=x.device, dtype=x.dtype)
        self._check(self.lib.rocblas_gemm_ex(*self.arguments(x, w, out), solution, 0))
        return out


trace = root.parent/'dsv4_prefill_tail_contract_20260917'
directory = trace/'trace-v2-mixed/trace'
meta = torch.load(directory/'layer-0-rank-0-metadata.pt', weights_only=False)['metadata']
record = json.loads((directory/'layer-0-rank-0-prepare_full_input.json').read_text())
sample_path = directory/record['values_file']
assert hashlib.sha256(sample_path.read_bytes()).hexdigest() == record['file_sha256']
raw = torch.load(sample_path, weights_only=False)['samples']
indices = [meta['sampled_rows'].index(i) for i in range(64)]
seed = raw[indices].contiguous().view(getattr(torch, record['dtype'].removeprefix('torch.'))).cuda()
weight_path = repo/'.agents/experiments/dsv4_input_identity_20260914/trace-B1/layer_0_rank_0_projection_wqkv_a.pt'
weight = torch.load(weight_path, weights_only=True).cuda()
digest = lambda v: hashlib.sha256(v.contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
expected = json.loads((directory/'layer-0-rank-0-projection_wqkv_a.json').read_text())['sha256']
assert digest(weight) == expected and weight.shape == (1536, 4096)
report = dict(status='running', scope=__doc__, weight_sha256=expected, input_sha256=digest(seed),
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    index_contract='https://github.com/ROCm/rocm-libraries/blob/8d1ae90eff7d022f26019ec55b2ec6a7674b3112/projects/rocblas/library/src/include/tensile_host.hpp#L50-L64',
    solution_lists={}, common_solutions=[], screen=[], validation=[], service_changed=False)


def save():
    target.write_text(json.dumps(report, indent=2)+'\n')


def compare(a, b):
    assert a.shape == b.shape and torch.isfinite(a).all() and torch.isfinite(b).all()
    diff = (a.double()-b.double()).abs()
    return dict(exact=torch.equal(a, b), different=int((a != b).sum()),
                max_abs=float(diff.max()), mean_abs=float(diff.mean()))


def time_graph(call):
    for _ in range(3):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(5):
            call()
    times = []
    for _ in range(6):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end)/5)
    return statistics.median(times)


fixed = Fixed()
save()
try:
    report['rocblas_library'] = str(fixed.path)
    report['rocblas_library_sha256'] = hashlib.sha256(fixed.path.read_bytes()).hexdigest()
    shapes = (1, 64, 65, 8192, 32576)
    for m in shapes:
        x = seed.repeat((m+63)//64, 1)[:m].contiguous()
        report['solution_lists'][str(m)] = fixed.solutions(x, weight)
        save()
    common = sorted(set.intersection(*(set(v) for v in report['solution_lists'].values())))
    report['common_solutions'] = common
    # A bounded initial screen, not an exhaustive best-algorithm claim.
    report['screen_limit'] = 32
    x = seed.repeat(509, 1).contiguous()
    out = torch.empty((len(x), 1536), device=x.device, dtype=x.dtype)
    report['torch_ms'] = time_graph(lambda: torch.mm(x, weight.t(), out=out))
    count = min(32, len(common))
    chosen = [common[(i*(len(common)-1))//max(1, count-1)] for i in range(count)]
    report['screen_indices'] = chosen
    for solution in chosen:
        ref = fixed.run(seed, weight, solution)
        y = fixed.run(x, weight, solution)
        reverse = x.flip(0).contiguous()
        yp = fixed.run(reverse, weight, solution)
        item = dict(solution=solution, shape=compare(y[:64], ref), row_order=compare(yp.flip(0), y))
        item['ms'] = time_graph(lambda: fixed.run(x, weight, solution, out))
        report['screen'].append(item)
        save()
        print('SCREEN', item, flush=True)
    passing = [r for r in report['screen'] if r['shape']['exact'] and r['row_order']['exact']]
    selected = sorted(passing, key=lambda v: v['ms'])[:4]
    for item in selected:
        solution = item['solution']
        for mutation in range(3):
            small = (seed.float()*(1-mutation*.011)).bfloat16().contiguous()
            base = fixed.run(small, weight, solution)
            fp64 = small.double() @ weight.double().t()
            for m in shapes:
                rows = small.repeat((m+63)//64, 1)[:m].contiguous()
                y = fixed.run(rows, weight, solution)
                permutation = torch.randperm(m, device='cuda')
                yp = fixed.run(rows[permutation].contiguous(), weight, solution)
                entry = dict(solution=solution, mutation=mutation, m=m,
                    shape=compare(y[:min(m,64)], base[:min(m,64)]),
                    row_order=compare(yp, y[permutation]), fp64=compare(y[:min(m,64)], fp64[:min(m,64)]))
                report['validation'].append(entry)
                save()
        replay_x = seed.repeat(128, 1).contiguous()
        replay_out = torch.empty((8192, 1536), device=seed.device, dtype=seed.dtype)
        eager_out = torch.empty_like(replay_out)
        fixed.run(replay_x, weight, solution, replay_out)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            fixed.run(replay_x, weight, solution, replay_out)
        successes = 0
        for iteration in range(100):
            replay_x.copy_((seed.roll(iteration % 64, 0).float()*(1-iteration*.0001)).bfloat16().repeat(128, 1))
            fixed.run(replay_x, weight, solution, eager_out)
            graph.replay()
            assert torch.equal(replay_out, eager_out), (solution, iteration)
            successes += 1
        report.setdefault('mutation_graph_checks', []).append(dict(solution=solution, exact=successes))
        save()
    report['status'] = 'complete'
    save()
except BaseException:
    report['status'] = 'failed'
    report['error'] = traceback.format_exc()
    save()
    raise
finally:
    fixed.close()
