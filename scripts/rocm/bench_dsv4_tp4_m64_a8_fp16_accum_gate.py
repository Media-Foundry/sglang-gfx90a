#!/usr/bin/env python3
"""M64 A8 FP16-local-accumulator gate oracle; no production selector."""

import argparse
import statistics

import torch

from sglang.kernels.ops.moe.gfx90a_fp4_expert_gemv import (
    _jit_gate_up_grouped_dpp,
    _jit_gate_up_grouped_dpp_fp16_accum,
)
from sglang.kernels.ops.quantization.int8_kernel import per_token_group_quant_int8

E, M, T, H, I, W, LDS = 256, 64, 6, 4096, 512, 8, 2


def reconstruct(counts):
    rows = [[] for _ in range(M)]
    for expert in torch.argsort(counts, descending=True).tolist():
        for _ in range(int(counts[expert])):
            choices = [t for t in range(M) if len(rows[t]) < T and expert not in rows[t]]
            token = min(choices, key=lambda t: (len(rows[t]), t))
            rows[token].append(expert)
    return torch.tensor(rows, dtype=torch.int32)


def metadata(topk, assignments):
    buckets = [[] for _ in range(E)]
    for token, row in enumerate(topk.tolist()):
        for slot, expert in enumerate(row):
            buckets[expert].append((slot << 24) | token)
    ids, experts = [], []
    sentinel = (T << 24) | M
    for expert, bucket in enumerate(buckets):
        for off in range(0, len(bucket), assignments):
            block = bucket[off : off + assignments]
            ids += block + [sentinel] * (assignments - len(block)); experts.append(expert)
    dev = torch.device("cuda")
    return (torch.tensor(ids, dtype=torch.int32, device=dev),
            torch.tensor(experts, dtype=torch.int32, device=dev),
            torch.tensor([len(ids), M], dtype=torch.int32, device=dev))


def timed(fn, warmup, iters):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); a,b=torch.cuda.Event(True),torch.cuda.Event(True);a.record()
    for _ in range(iters): fn()
    b.record();b.synchronize();return a.elapsed_time(b)*1000/iters


def main():
    ap=argparse.ArgumentParser();ap.add_argument("--recorder",default="/tmp/expert_distribution_recorder_1788072257.651073.pt")
    ap.add_argument("--pass-index",type=int,default=20);ap.add_argument("--layer",type=int,default=34)
    ap.add_argument("--blocks",type=int,nargs="+",default=[416,832,1040,2080]);ap.add_argument("--rows",type=int,nargs="+",default=[1,2])
    ap.add_argument("--rounds",type=int,default=7);ap.add_argument("--warmup",type=int,default=10);ap.add_argument("--iterations",type=int,default=50);ap.add_argument("--mutations",type=int,default=100)
    args=ap.parse_args()
    if not torch.version.hip or torch.cuda.get_device_properties(0).gcnArchName.split(":",1)[0]!="gfx90a":raise RuntimeError("gfx90a required")
    raw=torch.load(args.recorder,map_location="cpu",weights_only=False)["logical_count"][args.pass_index,args.layer]
    if torch.any(raw.remainder(4)!=0):raise RuntimeError("not TP4 recorder")
    topk=reconstruct(raw//4); a4=metadata(topk,4); a8=metadata(topk,8)
    print(f"a4_blocks={a4[1].numel()} a8_blocks={a8[1].numel()}")
    torch.manual_seed(83);dev=torch.device("cuda")
    x=torch.randn((M,H),dtype=torch.bfloat16,device=dev);xq,xs=per_token_group_quant_int8(x,32)
    weight=torch.randint(0,256,(E,2*I,H//2),dtype=torch.uint8,device=dev);scale=torch.full((E,2*I,H//32),127,dtype=torch.uint8,device=dev)
    ref=torch.empty((M,T,I),dtype=torch.bfloat16,device=dev);out=torch.empty_like(ref)
    mod_a=_jit_gate_up_grouped_dpp(E,M,T,I,H,4,2,W,2080,LDS)
    def run_a():mod_a.run(xq,xs,weight,scale,*a4,ref,10.0)
    candidates={}
    for rows in args.rows:
      for blocks in args.blocks:
        mod=_jit_gate_up_grouped_dpp_fp16_accum(E,M,T,I,H,8,rows,W,blocks,LDS)
        candidates[(rows,blocks)]=lambda mod=mod:mod.run(xq,xs,weight,scale,*a8,out,10.0)
    run_a();torch.cuda.synchronize()
    worst={k:{"max":0.0,"rel":0.0,"cos":1.0} for k in candidates}
    for mutation in range(args.mutations):
      xq.add_((mutation%7)+1);run_a();torch.cuda.synchronize();rf=ref.float();rn=torch.linalg.vector_norm(rf)
      for k,fn in candidates.items():
        fn();torch.cuda.synchronize();d=out.float()-rf
        worst[k]["max"]=max(worst[k]["max"],float(d.abs().max()))
        worst[k]["rel"]=max(worst[k]["rel"],float(torch.linalg.vector_norm(d)/rn))
        worst[k]["cos"]=min(worst[k]["cos"],float(torch.nn.functional.cosine_similarity(out.float().flatten(),rf.flatten(),dim=0)))
    for k in candidates:print("ERROR",k,worst[k])
    for k,fn in candidates.items():
      aa=[];bb=[]
      for _ in range(args.rounds):
        aa.append(timed(run_a,args.warmup,args.iterations));bb.append(timed(fn,args.warmup,args.iterations));bb.append(timed(fn,args.warmup,args.iterations));aa.append(timed(run_a,args.warmup,args.iterations))
      am,bm=statistics.median(aa),statistics.median(bb);print(f"ABBA rows={k[0]} blocks={k[1]} A_us={am:.3f} B_us={bm:.3f} delta={(bm/am-1)*100:+.2f}%")


if __name__=="__main__":main()
