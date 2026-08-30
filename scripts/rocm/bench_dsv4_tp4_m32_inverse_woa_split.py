#!/usr/bin/env python3
"""Split production inverse-RoPE and TP4 M32 wo_a graph costs."""

from __future__ import annotations
import argparse, os, statistics
import torch
import torch.distributed as dist

from sglang.kernels.ops.attention.dsv4.elementwise import fused_rope_inplace
from sglang.kernels.ops.quantization.gfx90a_bf16_gemv import (
    _jit_gfx90a_bf16_grouped_gemv_module,
    gfx90a_wave64_bf16_grouped_gemv,
)

M,G,D,R,ROPE=32,2,4096,1024,64

def cap(fn):
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out=fn()
    return g,out

def rankmax(g,iters,world):
    dist.barrier(); a=torch.cuda.Event(True); b=torch.cuda.Event(True); a.record()
    for _ in range(iters): g.replay()
    b.record();b.synchronize(); us=a.elapsed_time(b)*1000/iters
    vals=[None]*world;dist.all_gather_object(vals,us);return max(map(float,vals))

def trim(v): return statistics.mean(sorted(v)[1:-1])

def main():
    p=argparse.ArgumentParser();p.add_argument('--dump',default='/tmp/dsv4_ffn_dump.f3ZQ89')
    p.add_argument('--mutations',type=int,default=100);p.add_argument('--replays',type=int,default=1000)
    p.add_argument('--rounds',type=int,default=7);p.add_argument('--iters',type=int,default=300)
    a=p.parse_args(); lr=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(lr)
    dist.init_process_group('gloo');rank=dist.get_rank();world=dist.get_world_size()
    if world!=4: raise RuntimeError('requires four ranks')
    ps=[os.path.join(a.dump,f'layer_20_rank_{2*rank+i}_attn_inverse_rope.pt') for i in (0,1)]
    x=torch.cat([torch.load(q,map_location='cpu',weights_only=False) for q in ps],1).reshape(M,G,D).cuda().contiguous()
    base=x.clone(); mut=torch.linspace(-1,1,x.numel(),device='cuda').view_as(x).to(torch.bfloat16)
    gen=torch.Generator(device='cuda').manual_seed(44000+rank)
    w=torch.randn((G,R,D),generator=gen,device='cuda',dtype=torch.bfloat16).mul_(1/64)
    # Identity frequency keeps in-place replay idempotent. It exercises the
    # exact production HIP/Triton inverse-RoPE kernel without a graph copy.
    freqs=torch.ones((1,ROPE//2),device='cuda',dtype=torch.complex64)
    pos=torch.zeros((M,),device='cuda',dtype=torch.int64)
    mid_alias=torch.empty((M,G,R),device='cuda',dtype=torch.bfloat16)

    def prod_woa(inp):
        out=gfx90a_wave64_bf16_grouped_gemv(inp,w)
        return out if out is not None else torch.einsum('mgd,grd->mgr',inp,w)
    selector= gfx90a_wave64_bf16_grouped_gemv(x,w)
    selector_name='wave64' if selector is not None else 'einsum_fallback'
    rawmod=_jit_gfx90a_bf16_grouped_gemv_module(M)
    rawout=torch.empty((M,G,R),device='cuda',dtype=torch.bfloat16)

    def A(): fused_rope_inplace(x[...,-ROPE:],None,freqs,pos,inverse=True); return prod_woa(x)
    def B(): return prod_woa(x)
    def C(): return mid_alias
    def W(): rawmod.run(x,w,rawout); return rawout
    x.copy_(base); mid_alias.copy_(B()); torch.cuda.synchronize()
    ga,oa=cap(A);gb,ob=cap(B);gc,oc=cap(C);gw,ow=cap(W)

    raw_exact_all=True; raw_max=0.0
    for i in range(a.mutations):
        x.copy_(base).add_(mut,alpha=((i*1543+17)%2047-1023)/32768.0)
        mid_alias.copy_(B());ga.replay();gb.replay();gc.replay();gw.replay();torch.cuda.synchronize()
        if not (torch.equal(oa,ob) and torch.equal(oa,oc)): raise RuntimeError(f'mutation {i} production mismatch')
        raw_exact_all &= torch.equal(ob,ow);raw_max=max(raw_max,float((ob.float()-ow.float()).abs().max()))
    if rank==0: print(f'CORRECTNESS mutations={a.mutations} production_exact=True raw_wave64_exact={raw_exact_all} raw_wave64_max_abs={raw_max} selector={selector_name}')
    x.copy_(base);mid_alias.copy_(B());ga.replay();gb.replay();gc.replay();torch.cuda.synchronize()
    for i in range(a.replays):
        ga.replay();gb.replay();gc.replay()
        if (i+1)%100==0:
            torch.cuda.synchronize()
            if not (torch.equal(oa,ob) and torch.equal(oa,oc)): raise RuntimeError(f'replay {i+1} mismatch')
    vals={k:[] for k in 'ABCW'}; gs={'A':ga,'B':gb,'C':gc,'W':gw}
    for _ in range(a.rounds):
        for k in ('A','B','C','W','W','C','B','A'): vals[k].append(rankmax(gs[k],a.iters,world))
    if rank==0:
        t={k:trim(v) for k,v in vals.items()}
        for k in vals: print(f'RESULT profile={k} samples_us={",".join(f"{z:.3f}" for z in vals[k])} trimmed_rankmax_us={t[k]:.3f}')
        print(f'SPLIT inverse_rope_us={t["A"]-t["B"]:.3f} production_woa_us={t["B"]-t["C"]:.3f} floor_us={t["C"]:.3f} forced_wave64_us={t["W"]-t["C"]:.3f} selector={selector_name}')

if __name__=='__main__': main()
