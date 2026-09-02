#!/usr/bin/env python3
"""Full raw-FP4 expansion + variable-M CK MoE at production shapes."""

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_bf16_batched_moe import gfx90a_bf16_ck_moe

E,T,H,I=256,6,4096,512


def time_ms(fn, n=3):
    for _ in range(2): fn()
    torch.cuda.synchronize();a=torch.cuda.Event(True);b=torch.cuda.Event(True)
    a.record()
    for _ in range(n):fn()
    b.record();b.synchronize();return a.elapsed_time(b)/n


def routes(kind, device, m):
    if kind == "balanced":
        ids=torch.arange(m*T,device=device,dtype=torch.int32).reshape(m,T)%E
    else:
        # A deliberately difficult but valid distribution: three hot experts
        # plus three diverse experts per token.
        hot=torch.tensor([0,1,2],device=device,dtype=torch.int32).expand(m,3)
        tail=(torch.arange(m*3,device=device,dtype=torch.int32).reshape(m,3)%253)+3
        ids=torch.cat((hot,tail),dim=1)
    return ids,torch.rand((m,T),device=device,dtype=torch.float32)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--m",type=int,default=8192)
    parser.add_argument("--blocks",type=int,nargs="+",default=[832,1248,1664])
    args=parser.parse_args();m=args.m
    d=torch.device("cuda");torch.manual_seed(20260902)
    x=torch.randn((m,H),device=d,dtype=torch.bfloat16)
    w13=torch.randint(0,256,(E,2*I,H//2),device=d,dtype=torch.uint8)
    s13=torch.randint(118,132,(E,2*I,H//32),device=d,dtype=torch.uint8)
    w2=torch.randint(0,256,(E,H,I//2),device=d,dtype=torch.uint8)
    s2=torch.randint(118,132,(E,H,I//32),device=d,dtype=torch.uint8)
    for kind in ("balanced","skewed"):
        ids,tw=routes(kind,d,m);out=torch.empty((m,H),device=d,dtype=torch.bfloat16)
        for blocks in args.blocks:
            fn=lambda:gfx90a_bf16_ck_moe(x,ids,tw,w13,s13,w2,s2,out=out,blocks=blocks)
            fn();torch.cuda.synchronize();witness=out.clone();fn();torch.cuda.synchronize()
            if not torch.equal(witness,out):raise RuntimeError(f"{kind}: replay unstable")
            samples=[time_ms(fn) for _ in range(5)]
            print(f"m={m} kind={kind} blocks={blocks} median_ms={statistics.median(samples):.3f} samples={samples}")


if __name__=="__main__":main()
