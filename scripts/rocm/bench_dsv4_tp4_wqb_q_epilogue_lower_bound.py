#!/usr/bin/env python3
"""TP4/M32 lower bound for a hypothetical free wq_b Q norm/RoPE epilogue."""

from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import torch

from sglang.kernels.ops.attention.fused_qk_norm_rope_store import (
    fused_qk_norm_rope_swa_store,
)


def time_us(graph, iters):
    a=torch.cuda.Event(enable_timing=True); b=torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters): graph.replay()
    b.record(); b.synchronize(); return a.elapsed_time(b)*1000/iters


def trim(v): return statistics.mean(sorted(v)[1:-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump",type=Path,default=Path("/tmp/dsv4_ffn_dump.f3ZQ89"))
    p.add_argument("--mutations",type=int,default=100)
    p.add_argument("--replays",type=int,default=1000)
    p.add_argument("--rounds",type=int,default=7)
    p.add_argument("--iters",type=int,default=500)
    a=p.parse_args()
    real_q=torch.load(a.dump/"layer_20_rank_0_q.pt",map_location="cpu",weights_only=False)
    positions=torch.load(a.dump/"layer_20_rank_0_positions.pt",map_location="cpu",weights_only=False).cuda()
    if real_q.shape!=(32,8,512): raise RuntimeError(real_q.shape)
    # TP8 dump values, tiled only across the head axis to the exact TP4 H16 shape.
    q_source=real_q.repeat(1,2,1).reshape(32,8192).contiguous().cuda()
    kv_source=real_q[:,0,:].contiguous().cuda()
    kv_weight=torch.linspace(.75,1.25,512,dtype=torch.bfloat16,device="cuda")
    max_pos=max(1024,int(positions.max())+1); d=torch.arange(32,device="cuda",dtype=torch.float32)
    theta=positions.new_tensor(160000,dtype=torch.float32)
    inv=torch.pow(theta,-2*d/64); t=torch.arange(max_pos,device="cuda",dtype=torch.float32)
    phase=t[:,None]*inv[None,:]; cos=phase.cos(); sin=phase.sin()
    loc=torch.arange(32,dtype=torch.int32,device="cuda")
    q_a=q_source.clone(); q_b=q_source.clone(); kv_a=kv_source.clone();kv_b=kv_source.clone()
    out_a=torch.empty((32,16,512),dtype=torch.bfloat16,device="cuda")
    out_b=torch.empty_like(out_a); out_b.fill_(13)
    cache_a=torch.empty((64,512),dtype=torch.bfloat16,device="cuda")
    cache_b=torch.empty_like(cache_a)

    def run_a():
        kv_a.copy_(kv_source)
        return fused_qk_norm_rope_swa_store(q_a,kv_a,None,kv_weight,1e-6,1e-6,64,cos,sin,positions,cache_a,loc,1,out_a,torch.bfloat16,True)
    def run_b():
        kv_b.copy_(kv_source)
        return fused_qk_norm_rope_swa_store(q_b,kv_b,None,kv_weight,1e-6,1e-6,64,cos,sin,positions,cache_b,loc,1,out_b,torch.bfloat16,True,_oracle_q_off=True)

    run_a(); frozen_q=out_a.clone(); out_b.copy_(frozen_q); run_b();torch.cuda.synchronize()
    torch.testing.assert_close(kv_b,kv_a,rtol=0,atol=0);torch.testing.assert_close(cache_b[:32],cache_a[:32],rtol=0,atol=0)
    torch.testing.assert_close(out_b,frozen_q,rtol=0,atol=0)
    mutation=torch.empty_like(q_source)
    for i in range(a.mutations):
        mutation.normal_(); q_source.add_(mutation,alpha=1/4096);q_a.copy_(q_source);q_b.copy_(q_source)
        kv_source.copy_(q_source.view(32,16,512)[:,0,:]);run_a(); expected=out_a.clone();out_b.copy_(expected);run_b();torch.cuda.synchronize()
        torch.testing.assert_close(kv_b,kv_a,rtol=0,atol=0);torch.testing.assert_close(cache_b[:32],cache_a[:32],rtol=0,atol=0)
        torch.testing.assert_close(out_a,expected,rtol=0,atol=0);torch.testing.assert_close(out_b,expected,rtol=0,atol=0)
    print(f"CORRECTNESS mutations={a.mutations} kv_cache_exact=True q_free_oracle_immutable=True")
    frozen_q=out_a.clone();out_b.copy_(frozen_q)
    ga=torch.cuda.CUDAGraph();gb=torch.cuda.CUDAGraph()
    with torch.cuda.graph(ga): run_a()
    with torch.cuda.graph(gb): run_b()
    for _ in range(a.replays): ga.replay();gb.replay()
    torch.cuda.synchronize();torch.testing.assert_close(kv_b,kv_a,rtol=0,atol=0);torch.testing.assert_close(cache_b[:32],cache_a[:32],rtol=0,atol=0);torch.testing.assert_close(out_b,frozen_q,rtol=0,atol=0)
    print(f"CORRECTNESS graph_replays={a.replays} kv_cache_exact=True q_free_oracle_immutable=True")
    for _ in range(20):ga.replay();gb.replay()
    vals={'A':[],'B':[]}
    for _ in range(a.rounds):
        for n,g in (('A',ga),('B',gb),('B',gb),('A',ga)):vals[n].append(time_us(g,a.iters))
    aa,bb=trim(vals['A']),trim(vals['B'])
    print(f"RESULT qkv_us={aa:.3f} kv_only_us={bb:.3f} removable_q_us={aa-bb:.3f} passes_5us={aa-bb>=5}")
    print(f"BYTES raw_q={q_source.numel()*2} q_out={out_a.numel()*2} kv={kv_source.numel()*2} cache_compared={cache_a[:32].numel()*2}")

if __name__=='__main__':main()
