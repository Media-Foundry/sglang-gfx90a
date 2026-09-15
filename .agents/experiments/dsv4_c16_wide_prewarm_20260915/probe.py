"""Compile-only wide-indexer priming and first-use replay, isolated GCD4.

No model loading or fabricated KV launch during prime. Fresh private compiler
caches distinguish cold compilation from reuse across a process restart.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--mode',choices=['prime','warm','cold'],required=True)
args=p.parse_args()
root=Path(__file__).resolve().parent/'v2'
root.mkdir(exist_ok=True)
cache=root/('cold-cache' if args.mode=='cold' else 'primed-cache')
if args.mode in ('prime','cold'):cache.mkdir(exist_ok=False)
else:assert cache.is_dir() and (root/'prime.json').exists()
target=root/f'{args.mode}.json';assert not target.exists()
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
owners=json.loads(subprocess.check_output(['amd-smi','process','--json']))
assert not any(isinstance(q.get('process_info'),dict) for g in owners for q in g.get('process_list',[]))
os.environ['TRITON_CACHE_DIR']=str(cache)
os.environ.pop('TRITON_ALWAYS_COMPILE',None)
os.environ.pop('TRITON_OVERRIDE_DIR',None)
import torch
import triton
import triton.language as tl
from triton.runtime.jit import MockTensor
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import reuse_runtime_m as kernel

torch.cuda.set_device(0)
launches=[]
triton.knobs.runtime.launch_enter_hook=lambda *a,**k: launches.append(time.perf_counter())
signatures=[(4096,64,64),(8192,128,128),(8192,128,131)]
def tail(m,w,n,pt):return (m,w,n,pt,16,16,16,tl.bfloat16,tl.float8e4nv)
def artifact(c):
    return dict(hash=c.hash,hsaco_sha256=hashlib.sha256(c.asm['hsaco']).hexdigest(),
                registers=getattr(c,'n_regs',None),spills=getattr(c,'n_spills',None),lds=c.metadata.shared)
records=[]
if args.mode=='prime':
    for w,n,pt in signatures:
        mem=torch.cuda.memory_allocated();count=len(launches);start=time.perf_counter()
        c=kernel.warmup(*[MockTensor(t) for t in
            (torch.uint8,torch.uint8,torch.float32,torch.int32,torch.int32,torch.float32)],
            *tail(32768,w,n,pt),grid=(2048,w//16),num_warps=4)
        seconds=time.perf_counter()-start
        assert len(launches)==count, 'Compile-only warmup unexpectedly launched a kernel'
        assert torch.cuda.memory_allocated()==mem
        records.append(dict(width=w,pages=n,page_stride=pt,compile_s=seconds,
                            tensor_allocated_delta=0,kernel_launches=0,**artifact(c)))
        print('PRIMED',records[-1],flush=True)
else:
    for index,(w,n,pt) in enumerate(signatures):
        torch.manual_seed(1729+index)
        m=8192 if w==4096 else 32767
        q=torch.randn((m,1,64,128),device='cuda').to(torch.float8_e4m3fn)
        weights=torch.randn((m,64),device='cuda')
        lengths=(513+torch.arange(m,device='cuda')%(w-512)).int()
        # All rows use valid shared pages. Per-position FP8 data and FP32 scales
        # are packed/preshuffled by the same store used in the serving kernel.
        pages_storage=torch.empty((m,pt),device='cuda',dtype=torch.int32)
        pages=pages_storage[:,:n];pages.copy_(torch.arange(n,device='cuda'))
        cache_tensor=torch.empty((n,64,1,132),device='cuda',dtype=torch.uint8)
        from sglang.kernels.ops.kvcache.triton_store_cache import triton_fused_store_indexer
        values=torch.randn((n*64,128),device='cuda')
        loc=torch.arange(n*64,device='cuda',dtype=torch.int32)
        triton_fused_store_indexer(values,cache_tensor.view(n,8448),loc,64)
        out=torch.empty((m,w),device='cuda')
        tensors=(q.view(torch.uint8),cache_tensor,weights,lengths,pages,out)
        assert all(t.data_ptr()%16==0 for t in tensors)
        torch.cuda.synchronize();count=len(launches);start=time.perf_counter()
        c=kernel[(triton.cdiv(m,16),w//16)](*tensors,*tail(m,w,n,pt),num_warps=4)
        torch.cuda.synchronize();first=time.perf_counter()-start
        assert len(launches)==count+1
        digest=lambda t:hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
        output_hash=digest(out)
        assert torch.isfinite(out).all()
        assert torch.count_nonzero(torch.where(torch.arange(w,device='cuda')[None,:]>=lengths[:,None],out,0))==0
        inputs={name:digest(t) for name,t in zip(('q','cache','weights','lengths','pages'),tensors[:-1])}
        samples=[]
        for _ in range(3):
            start=time.perf_counter();d=kernel[(triton.cdiv(m,16),w//16)](*tensors,*tail(m,w,n,pt),num_warps=4)
            torch.cuda.synchronize();samples.append(time.perf_counter()-start)
            assert d.hash==c.hash
        assert digest(out)==output_hash
        records.append(dict(m=m,width=w,pages=n,page_stride=pt,first_use_s=first,
            repeated_use_s=samples,input_hashes=inputs,output_sha256=output_hash,**artifact(c)))
        print(args.mode.upper(),{k:v for k,v in records[-1].items() if k!='input_hashes'},flush=True)
        del q,weights,lengths,pages,pages_storage,cache_tensor,values,loc,out,tensors
        torch.cuda.empty_cache()
result=dict(mode=args.mode,records=records,cache=str(cache),gpu=4,
    dtype_contract=dict(fp8='float8_e4m3fn',shuffle=16,dot='bf16'),
    source_sha256=hashlib.sha256(Path(kernel.fn.__code__.co_filename).read_bytes()).hexdigest(),
    scope=__doc__)
target.write_text(json.dumps(result,indent=2)+'\n')
