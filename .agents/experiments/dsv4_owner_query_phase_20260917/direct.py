"""Experiment-local per-stream Lt plans; no environment or global BLAS changes."""
import ctypes as C
import hashlib
import json
from pathlib import Path

import torch


class Direct:
    def __init__(self, manifest):
        record=json.loads(Path(manifest).read_text())
        for path_key,hash_key in [('output','output_sha256'),('source','source_sha256'),('hipblaslt','hipblaslt_sha256')]:
            assert hashlib.sha256(Path(record[path_key]).read_bytes()).hexdigest()==record[hash_key]
        self.record=record
        self.lib=C.CDLL(record['output'])
        ptr,size,integer=C.c_void_p,C.c_size_t,C.c_int
        self.lib.dsv4_lt_create.argtypes=[integer,integer,integer,integer,C.POINTER(ptr),C.POINTER(size)]
        self.lib.dsv4_lt_run.argtypes=[ptr,ptr,ptr,ptr,ptr,size,ptr]
        self.lib.dsv4_lt_destroy.argtypes=[ptr]
        self.lib.dsv4_lt_error.restype=C.c_char_p
        self.device=torch.cuda.current_device()
        self.stream=torch.cuda.current_stream()
        self.plans={}
        self.workspace=None

    def check(self,status):
        if status!=0:raise RuntimeError(self.lib.dsv4_lt_error().decode())

    def prepare(self,m,n,k):
        assert not torch.cuda.is_current_stream_capturing()
        key=(m,n,k)
        if key not in self.plans:
            plan=C.c_void_p();size=C.c_size_t()
            self.check(self.lib.dsv4_lt_create(m,n,k,self.record['direct_index'],C.byref(plan),C.byref(size)))
            self.plans[key]=(plan,size.value)
            if size.value and (self.workspace is None or len(self.workspace)<size.value):
                self.workspace=torch.empty(size.value,device=f'cuda:{self.device}',dtype=torch.uint8)
        return self.plans[key]

    def __call__(self,x,w,out=None):
        assert torch.cuda.current_device()==self.device
        assert torch.cuda.current_stream().cuda_stream==self.stream.cuda_stream
        assert x.device==w.device and x.device.index==self.device
        assert x.dtype==w.dtype==torch.bfloat16 and x.is_contiguous() and w.is_contiguous()
        m,k=x.shape;n=w.shape[0];assert w.shape[1]==k
        key=(m,n,k)
        plan,required=self.plans[key] if key in self.plans else self.prepare(*key)
        if out is None:out=torch.empty((m,n),device=x.device,dtype=x.dtype)
        assert out.shape==(m,n) and out.dtype==x.dtype and out.device==x.device and out.is_contiguous()
        workspace=self.workspace
        self.check(self.lib.dsv4_lt_run(plan,C.c_void_p(x.data_ptr()),C.c_void_p(w.data_ptr()),C.c_void_p(out.data_ptr()),
            C.c_void_p(workspace.data_ptr()) if workspace is not None else None,
            len(workspace) if workspace is not None else 0,C.c_void_p(self.stream.cuda_stream)))
        return out

    def close(self):
        self.stream.synchronize()
        for plan,_ in self.plans.values():self.lib.dsv4_lt_destroy(plan)
        self.plans.clear();self.workspace=None
