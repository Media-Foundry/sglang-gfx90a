"""Pair/quad query rows while retaining FP32 Fn and K1024 reduction tiles.

Component screen only. Repeated/scaled real sampled residual rows are synthetic
occupancy, not a fresh full-model capture. No production dispatch is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
from statistics import median

import torch
import triton
import triton.language as tl
from sglang.kernels.ops.layernorm.mhc import gfx90a_mhc_pre_mix_from_partials_triton


@triton.jit
def reuse(x,fn,partials,out,M,BM:tl.constexpr,EPS:tl.constexpr=1e-6):
    row=tl.program_id(1)*BM
    n=tl.program_id(0)+tl.arange(0,1)
    acc0=tl.zeros((1,),tl.float32)
    acc1=tl.zeros((1,),tl.float32)
    if BM==4:
        acc2=tl.zeros((1,),tl.float32)
        acc3=tl.zeros((1,),tl.float32)
    for start in tl.static_range(0,16384,1024):
        k=start+tl.arange(0,1024)
        w=tl.load(fn+n[:,None]*16384+k[None,:],mask=n[:,None]<24,other=0.)
        a0=tl.load(x+row*16384+k).to(tl.float32)
        a1=tl.load(x+(row+1)*16384+k,mask=row+1<M,other=0.).to(tl.float32)
        acc0+=tl.sum(w*a0[None,:],axis=1)
        acc1+=tl.sum(w*a1[None,:],axis=1)
        if BM==4:
            a2=tl.load(x+(row+2)*16384+k,mask=row+2<M,other=0.).to(tl.float32)
            a3=tl.load(x+(row+3)*16384+k,mask=row+3<M,other=0.).to(tl.float32)
            acc2+=tl.sum(w*a2[None,:],axis=1)
            acc3+=tl.sum(w*a3[None,:],axis=1)
    p=tl.arange(0,64)
    sq0=tl.sum(tl.load(partials+row*64+p),axis=0)
    sq1=tl.sum(tl.load(partials+(row+1)*64+p,mask=row+1<M,other=0.),axis=0)
    tl.store(out+row*24+n,acc0*tl.rsqrt(sq0/16384+EPS),mask=n<24)
    tl.store(out+(row+1)*24+n,acc1*tl.rsqrt(sq1/16384+EPS),mask=(n<24)&(row+1<M))
    if BM==4:
        sq2=tl.sum(tl.load(partials+(row+2)*64+p,mask=row+2<M,other=0.),axis=0)
        sq3=tl.sum(tl.load(partials+(row+3)*64+p,mask=row+3<M,other=0.),axis=0)
        tl.store(out+(row+2)*24+n,acc2*tl.rsqrt(sq2/16384+EPS),mask=(n<24)&(row+2<M))
        tl.store(out+(row+3)*24+n,acc3*tl.rsqrt(sq3/16384+EPS),mask=(n<24)&(row+3<M))


def measure(fn):
    a,b=torch.cuda.Event(True),torch.cuda.Event(True)
    a.record()
    for _ in range(5):fn()
    b.record();b.synchronize()
    return a.elapsed_time(b)/5


def compare(a,b):
    delta=a-b
    return dict(exact=torch.equal(a,b),mismatch=int((a!=b).sum()),
                max_abs=float(delta.abs().max()),
                relative_l2=float(torch.linalg.vector_norm(delta)/torch.linalg.vector_norm(a)),
                finite=bool(torch.isfinite(b).all()))


p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--mutations',type=int,default=3)
p.add_argument('--randomize-weight',action='store_true')
p.add_argument('--bm',type=int,nargs='+',default=[2,4],choices=[2,4])
p.add_argument('--sizes',type=int,nargs='+',default=[128,8192,32767,32768])
p.add_argument('--runtime',action='store_true')
args=p.parse_args();assert not args.output.exists()
if args.runtime:
    assert os.environ.get('SGLANG_DSV4_PREFILL_MIX_REUSE4')=='1' and args.bm==[4]
    from sglang.kernels.ops.layernorm.gfx90a_mhc_premix_reuse import _premix_reuse_kernel as reuse
    from sglang.srt.layers.dsv4_prefill_experiments import _mix_reuse
assert os.environ.get('HIP_VISIBLE_DEVICES')=='4'
os.environ['SGLANG_DSV4_GFX90A_MHC_BLOCK_K']='1024'
torch.manual_seed(20260915)
root=Path(__file__).resolve().parent
source=root.parent/'dsv4_input_identity_20260914'/'trace-B1'
path=source/'layer_0_rank_0_hc_ffn_fn.pt'
fn=torch.load(path,map_location='cuda',weights_only=True).contiguous()
fn_seed=fn.clone()
seed=torch.load(source/'layer_0_rank_0_ffn_mhc_residual.pt',map_location='cuda',weights_only=True)
result=dict(scope=__doc__,runtime=args.runtime,fn_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),results=[])
for m in args.sizes:
    x=seed.repeat(triton.cdiv(m,len(seed)),1,1)[:m].contiguous()
    x.mul_(torch.empty((m,1,1),dtype=x.dtype,device='cuda').uniform_(.8,1.2))
    partials=x.float().square().view(m,64,256).sum(-1)
    out=torch.empty((m,1,24),dtype=torch.float32,device='cuda')
    eps=1e-6
    def control():return gfx90a_mhc_pre_mix_from_partials_triton(x,fn,partials,eps)
    for bm in args.bm:
        eps=1e-6;fn.copy_(fn_seed)
        def candidate():
            global out
            if args.runtime:
                token=_mix_reuse.set(True)
                try:
                    out=gfx90a_mhc_pre_mix_from_partials_triton(x,fn,partials,eps)
                finally:
                    _mix_reuse.reset(token)
                return out
            reuse[(24,triton.cdiv(m,bm))](x,fn,partials,out,m,bm,eps,num_warps=1)
            return out
        for _ in range(3):control();candidate()
        initial=compare(control().clone(),candidate().clone())
        samples={'A':[],'B':[]}
        for _ in range(3):
            for name,func in [('A',control),('B',candidate),('B',candidate),('A',control)]:
                samples[name].append(measure(func))
        mutations=[]
        for mutation in range(args.mutations):
            if args.randomize_weight:
                fn.normal_(std=.01)
                x.normal_()
                eps=(1e-6,1e-5,1e-8)[mutation%3]
            else:
                x.mul_(torch.empty((m,1,1),dtype=x.dtype,device='cuda').uniform_(.97,1.03))
            partials.copy_(x.float().square().view(m,64,256).sum(-1))
            mutations.append(compare(control().clone(),candidate().clone()))
        ac=control().clone();bc=candidate().clone()
        order=torch.randperm(m,device='cuda');inverse=torch.argsort(order)
        x.copy_(x[order]);partials.copy_(partials[order])
        graph_exact=None
        if m==128:
            graph=torch.cuda.CUDAGraph()
            candidate();torch.cuda.synchronize()
            with torch.cuda.graph(graph):candidate()
            for _ in range(1000):graph.replay()
            torch.cuda.synchronize()
            graph_exact=torch.equal(control(),out)
        compiled=reuse[(24,triton.cdiv(m,bm))](x,fn,partials,out,m,bm,eps,num_warps=1)
        resources=dict(n_regs=getattr(compiled,'n_regs',None),n_spills=getattr(compiled,'n_spills',None),
                       shared_bytes=getattr(compiled.metadata,'shared',None))
        item=dict(m=m,bm=bm,initial=initial,mutations=mutations,samples_ms=samples,
                  randomize_weight=args.randomize_weight,resources=resources,graph_1000_exact=graph_exact,
                  median_ms={k:median(v) for k,v in samples.items()},
                  control_permutation=compare(ac,control()[inverse]),
                  candidate_permutation=compare(bc,candidate()[inverse]))
        item['speedup']=item['median_ms']['A']/item['median_ms']['B']
        result['results'].append(item)
        args.output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(item),flush=True)
    del x,partials,out,ac,bc,order,inverse
    torch.cuda.empty_cache()
