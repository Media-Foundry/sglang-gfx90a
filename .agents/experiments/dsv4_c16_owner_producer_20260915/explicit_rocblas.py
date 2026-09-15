"""Experimental per-object rocBLAS handle, never changes global BLAS selection.

For isolated ABI/stream tests only, not wired into a production model. Constants
and signature checked against /opt/rocm/include/rocblas/internal/rocblas-types.h
and rocblas-functions.h (rocblas_gemm_ex). No weights or dtype conversion.
"""
import ctypes as C
from pathlib import Path

import torch


class Linear:
    def __init__(self):
        if not torch.version.hip:
            raise RuntimeError('ROCm only')
        self.device = torch.cuda.current_device()
        self.path = Path('/opt/rocm/lib/librocblas.so').resolve(strict=True)
        self.lib = C.CDLL(str(self.path))
        ptr, integer = C.c_void_p, C.c_int
        self.lib.rocblas_create_handle.argtypes = [C.POINTER(ptr)]
        self.lib.rocblas_destroy_handle.argtypes = [ptr]
        self.lib.rocblas_set_stream.argtypes = [ptr, ptr]
        self.lib.rocblas_set_atomics_mode.argtypes = [ptr, integer]
        self.lib.rocblas_gemm_ex.argtypes = [
            ptr, integer, integer, integer, integer, integer,
            ptr, ptr, integer, integer, ptr, integer, integer,
            ptr, ptr, integer, integer, ptr, integer, integer,
            integer, integer, integer, C.c_uint32,
        ]
        self.handle = ptr()
        self._check(self.lib.rocblas_create_handle(C.byref(self.handle)))
        self._check(self.lib.rocblas_set_atomics_mode(self.handle, 0))
        self.alpha, self.beta = C.c_float(1.0), C.c_float(0.0)

    @staticmethod
    def _check(status):
        if status != 0:
            raise RuntimeError(f'rocBLAS status {status}')

    def __call__(self, x, w):
        assert self.handle.value and torch.cuda.current_device() == self.device
        assert x.ndim == w.ndim == 2 and x.shape[1] == w.shape[1]
        assert x.is_cuda and x.device == w.device and x.device.index == self.device
        assert x.dtype == w.dtype == torch.bfloat16
        assert x.is_contiguous() and w.is_contiguous()
        m, k = x.shape
        n = w.shape[0]
        assert max(m, n, k) < 2**31 and k > 0 and n > 0
        out = torch.empty((m, n), dtype=x.dtype, device=x.device)
        if m == 0:
            return out
        stream = torch.cuda.current_stream(x.device)
        self._check(self.lib.rocblas_set_stream(self.handle, C.c_void_p(stream.cuda_stream)))
        # Row-major Y=X W^T becomes column-major Y^T=W X^T.
        # 112=transpose,111=none,168=BF16,151=FP32,algo0=standard.
        self._check(self.lib.rocblas_gemm_ex(
            self.handle, 112, 111, n, m, k,
            C.byref(self.alpha), C.c_void_p(w.data_ptr()), 168, k,
            C.c_void_p(x.data_ptr()), 168, k,
            C.byref(self.beta), C.c_void_p(out.data_ptr()), 168, n,
            C.c_void_p(out.data_ptr()), 168, n, 151, 0, 0, 0,
        ))
        return out

    def close(self):
        if self.handle.value:
            self._check(self.lib.rocblas_destroy_handle(self.handle))
            self.handle = C.c_void_p()


def main():
    import argparse
    import hashlib
    import json
    import torch.nn.functional as F
    from safetensors import safe_open
    from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--production', action='store_true')
    args = parser.parse_args()
    assert not args.output.exists()
    root = Path(__file__).resolve().parent
    r = json.loads((root/'capture/oracle/report.json').read_text())
    path = root/'capture/oracle/inputs.pt'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == r['inputs_file_sha256']
    data = torch.load(path, map_location='cpu', weights_only=True)
    model = Path('/home/pc/models/modelscope')
    key = 'layers.20.attn.indexer.wq_b'
    idx = json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
    with safe_open(model/idx[key+'.weight'], framework='pt', device='cpu') as f:
        weight = (f.get_tensor(key+'.weight').float().reshape(64,128,8,128)
                  * f.get_tensor(key+'.scale').float()[:,None,:,None]).reshape(8192,1024).bfloat16()
    assert hashlib.sha256(weight.view(torch.uint8).numpy().tobytes()).hexdigest() == r['parameters']['wq_b']['weight']['sha256']
    _, ids, valid, _ = host_plan(r['extend_lens'], r['prefix_lens'], 0)
    assert valid.all()
    x, w = data['q_lora'].cuda(), weight.cuda()
    part = x.index_select(0, torch.from_numpy(ids).cuda())
    previous = torch.backends.cuda.preferred_blas_library()
    if args.production:
        from sglang.kernels.ops.attention.dsv4.gfx90a_rocblas_linear import Linear as ProductionLinear
        operator = ProductionLinear()
    else:
        operator = Linear()
    results = dict(diagnostic_only=True, library=str(operator.path), shapes=[],
                   production_wrapper=args.production,
                   source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    try:
        for xx in (part[:3], part[:17], part, x):
            torch.backends.cuda.preferred_blas_library('cublas')
            reference = F.linear(xx,w)
            torch.backends.cuda.preferred_blas_library(previous)
            actual = operator(xx,w)
            torch.cuda.synchronize()
            item = dict(rows=len(xx), exact=bool(torch.equal(reference,actual)),
                        max_abs=float((reference.float()-actual.float()).abs().max()))
            assert item['exact'], item
            alternate = torch.cuda.Stream()
            alternate.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(alternate):
                stream_output = operator(xx,w)
            alternate.synchronize()
            item['alternate_stream_exact'] = bool(torch.equal(reference,stream_output))
            assert item['alternate_stream_exact']
            assert torch.backends.cuda.preferred_blas_library() == previous
            results['shapes'].append(item)
        g = torch.cuda.CUDAGraph()
        for _ in range(3): operator(part,w)
        torch.cuda.synchronize()
        with torch.cuda.graph(g): output = operator(part,w)
        g.replay(); torch.cuda.synchronize(); first = output.clone()
        for _ in range(100): g.replay()
        torch.cuda.synchronize()
        results['graph100_exact'] = bool(torch.equal(first,output))
        assert results['graph100_exact']
        original_part = part.clone()
        results['graph_input_mutations_exact'] = 0
        for mutation in range(20):
            # Real rows, changed order and amplitude; same captured pointers.
            part.copy_(torch.roll(original_part,mutation+1,0) * (0.75+mutation/40))
            g.replay()
            torch.backends.cuda.preferred_blas_library('cublas')
            reference = F.linear(part,w)
            torch.backends.cuda.preferred_blas_library(previous)
            torch.cuda.synchronize()
            assert torch.equal(reference,output), mutation
            results['graph_input_mutations_exact'] += 1
        assert torch.backends.cuda.preferred_blas_library() == previous
        results['status'] = 'complete'
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(json.dumps(results,indent=2),flush=True)
    finally:
        torch.cuda.synchronize()
        operator.close()
        torch.backends.cuda.preferred_blas_library(previous)


if __name__ == '__main__': main()
