#!/usr/bin/env python3
"""ABBA shuffled versus logical-contiguous E8M0 scales for TP4 grouped MoE."""

from __future__ import annotations

import argparse
import statistics

import torch

from scripts.rocm.bench_dsv4_gfx90a_occupancy_bucket_oracle import (
    make_metadata,
    reconstruct_topk_from_counts,
)
from sglang.kernels.jit.utils import cache_once, load_jit, make_cpp_args
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8

E, M, T, I, H, N = 256, 32, 6, 512, 4096, 4096
A, R, W, G, D, LUT = 4, 2, 8, 2080, 832, 2


@cache_once
def gate(logical: bool):
    args = make_cpp_args(E, M, T, I, H, A, R, W, G, LUT, logical)
    return load_jit(
        "gfx90a_fp4_gate_scale_layout_oracle", *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_gate_row_prefetch_oracle.cuh"],
        cuda_wrappers=[("run", f"sglang::Gfx90aFp4ExpertGateRowPrefetchOracle<{args}>::run")],
        extra_cuda_cflags=["-O3"],
    )


@cache_once
def down(logical: bool):
    args = make_cpp_args(E, M, T, N, I, A, W, D, LUT, logical)
    return load_jit(
        "gfx90a_fp4_down_scale_layout_oracle", *args,
        cuda_files=["deepseek_v4/gfx90a_fp4_expert_down_row_prefetch_oracle.cuh"],
        cuda_wrappers=[
            ("run_partial", f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::run_partial"),
            ("reduce", f"sglang::Gfx90aFp4ExpertDownRowPrefetchOracle<{args}>::reduce"),
        ], extra_cuda_cflags=["-O3"],
    )


def gate_logical(s: torch.Tensor) -> torch.Tensor:
    # physical [E,N1,K1,klane,nlane,kpack,gate_up]
    return s.reshape(E, I // 16, (H // 32) // 8, 4, 16, 2, 2).permute(
        0, 6, 1, 4, 2, 5, 3
    ).contiguous().reshape(E, 2 * I, H // 32)


def down_logical(s: torch.Tensor) -> torch.Tensor:
    # physical [E,N1,K1,klane,nlane,kpack,npack]
    return s.reshape(E, N // 32, (I // 32) // 8, 4, 16, 2, 2).permute(
        0, 1, 6, 4, 2, 5, 3
    ).contiguous().reshape(E, N, I // 32)


def time_us(fn, iterations=20):
    for _ in range(5): fn()
    torch.cuda.synchronize(); a=torch.cuda.Event(True); b=torch.cuda.Event(True)
    a.record()
    for _ in range(iterations): fn()
    b.record(); b.synchronize(); return a.elapsed_time(b)*1000/iterations


def trim(v): return statistics.mean(sorted(v)[1:-1])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recorder", default="/tmp/expert_distribution_recorder_1787803355.1855972.pt")
    p.add_argument("--pass-index",type=int,default=37); p.add_argument("--layer",type=int,default=34)
    p.add_argument("--mutations",type=int,default=100); p.add_argument("--rounds",type=int,default=7)
    a=p.parse_args(); payload=torch.load(a.recorder,map_location="cpu",weights_only=False)
    counts=payload["logical_count"][a.pass_index,a.layer]//8
    md=make_metadata(reconstruct_topk_from_counts(counts).cuda(),assignments=A)
    torch.manual_seed(20260830)
    x=torch.randn((M,H),dtype=torch.bfloat16,device="cuda"); xq,xs=per_token_group_quant_int8(x,32)
    w13=torch.randint(0,256,(E,2*I,H//2),dtype=torch.uint8,device="cuda")
    s13=torch.randint(110,135,(E,2*I,H//32),dtype=torch.uint8,device="cuda")
    w2=torch.randint(0,256,(E,N,I//2),dtype=torch.uint8,device="cuda")
    s2=torch.randint(110,135,(E,N,I//32),dtype=torch.uint8,device="cuda")
    ls13=gate_logical(s13); ls2=down_logical(s2)
    print(f"MEMORY gate_bytes={ls13.numel()} down_bytes={ls2.numel()} per_layer_mib={(ls13.numel()+ls2.numel())/2**20:.3f} layers43_gib={(ls13.numel()+ls2.numel())*43/2**30:.3f}")
    tw=torch.rand((M,T),dtype=torch.float32,device="cuda")
    states={}
    for name,logical in (("A",False),("B",True)):
        st={"mid":torch.empty((M,T,I),dtype=torch.bfloat16,device="cuda"),"part":torch.empty((M,T,N),dtype=torch.float32,device="cuda"),"out":torch.empty((M,N),dtype=torch.bfloat16,device="cuda")}
        gm=gate(logical); dm=down(logical); gs=ls13 if logical else s13; ds=ls2 if logical else s2
        def g(gm=gm,gs=gs,st=st): gm.run(xq,xs,w13,gs,md.sorted_ids,md.sorted_experts,md.valid,st["mid"],10.0)
        def q(st=st): st["iq"],st["isc"]=per_token_group_quant_int8(st["mid"],32)
        def d(dm=dm,ds=ds,st=st): dm.run_partial(st["iq"],st["isc"],w2,ds,md.sorted_ids,md.sorted_experts,md.valid,tw,st["part"])
        def r(dm=dm,st=st): dm.reduce(st["part"],st["out"])
        def full(g=g,q=q,d=d,r=r): g();q();d();r()
        states[name]=(st,{"gate":g,"quant":q,"down":d,"reduce":r,"full":full})
    mut=torch.empty_like(x)
    for i in range(a.mutations):
        mut.normal_(); q,s=per_token_group_quant_int8(mut,32); xq.copy_(q);xs.copy_(s);tw.uniform_()
        for n in "AB": states[n][1]["full"]()
        torch.cuda.synchronize()
        for key in ("mid","iq","isc","part","out"):
            if not torch.equal(states["A"][0][key],states["B"][0][key]):
                raise RuntimeError(f"mutation={i} key={key} mismatch")
    print(f"CORRECTNESS mutations={a.mutations} all_exact=True")
    timings={k:{"A":[],"B":[]} for k in ("gate","quant","down","reduce","full")}
    for _ in range(a.rounds):
        for n in ("A","B","B","A"):
            for k in timings: timings[k][n].append(time_us(states[n][1][k]))
    for k,v in timings.items():
        aa,bb=trim(v["A"]),trim(v["B"]); print(f"RESULT stage={k} shuffled_us={aa:.3f} logical_us={bb:.3f} delta_us={bb-aa:.3f} gain_pct={(aa/bb-1)*100:.3f}")


if __name__=="__main__": main()
