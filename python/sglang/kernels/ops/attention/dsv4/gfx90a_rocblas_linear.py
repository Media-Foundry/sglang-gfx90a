"""Experimental per-object rocBLAS handle, never changes global BLAS selection.

Used only by the opt-in original-V4 owner-query producer. Constants
and signature checked against /opt/rocm/include/rocblas/internal/rocblas-types.h
and rocblas-functions.h (rocblas_gemm_ex). No weights or dtype conversion.
"""
import ctypes as C
import weakref
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
        self._finalize = weakref.finalize(self, self.lib.rocblas_destroy_handle, self.handle)
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
            self._check(self._finalize())
            self.handle = C.c_void_p()
