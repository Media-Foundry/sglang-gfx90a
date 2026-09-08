#!/usr/bin/env python3
"""Real layer0 repeated-prefix projection oracle; no production wiring."""
import torch
import argparse
import statistics
import triton
import triton.language as tl
from pathlib import Path


@triton.jit
def gemm(X,W,Y,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
         BM:tl.constexpr=16,BN:tl.constexpr=64,BK:tl.constexpr=64):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(tl.cdiv(K,BK)):
        kk=start*BK+k
        a=tl.load(X+m[:,None]*K+kk[None,:],m[:,None]<M,other=0)
        b=tl.load(W+n[None,:]*K+kk[:,None],n[None,:]<N,other=0)
        acc=tl.dot(a,b,acc)
    tl.store(Y+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bm',type=int,default=16)
    parser.add_argument('--bn',type=int,default=64)
    parser.add_argument('--bk',type=int,choices=(64,128,256),default=64)
    parser.add_argument('--warps',type=int,default=4)
    parser.add_argument('--stages',type=int,default=2)
    parser.add_argument('--request-bmm',action='store_true',
                        help='Use shared-weight batched GEMM over16 equal46-token prefixes')
    args=parser.parse_args()
    print('config',vars(args),flush=True)
    p=Path('/tmp/dsv4_tp8_slot_layer0_20260908')
    x=torch.load(p/'layer_0_rank_0_attn_norm.pt',weights_only=True).cuda()
    w=torch.load(p/'layer_0_rank_0_projection_wqkv_a.pt',weights_only=True).cuda()
    y=torch.empty((x.shape[0],w.shape[0]),device='cuda',dtype=torch.bfloat16)
    def candidate():
        if args.request_bmm:
            torch.bmm(x.view(16,46,4096),w.t().expand(16,-1,-1),out=y.view(16,46,1536))
        else:
            gemm[(triton.cdiv(x.shape[0],args.bm),triton.cdiv(w.shape[0],args.bn))](x,w,y,x.shape[0],w.shape[0],x.shape[1],BM=args.bm,BN=args.bn,BK=args.bk,num_warps=args.warps,num_stages=args.stages)
    candidate();torch.cuda.synchronize()
    # FP64 is diagnostic only; use the first46 rows to bound temporary memory.
    reference=(x[:46].double()@w.double().t()).bfloat16()
    print('fp64_reference_max_abs',(y[:46].float()-reference.float()).abs().max().item(),flush=True)
    for name,z in [('torch',torch.nn.functional.linear(x,w)),('candidate',y)]:
        q=z.float().view(16,46,-1);d=q-q[:1]
        print(name,'max_row_delta',d.abs().max().item(),'exact_rows',(d==0).all(-1).sum().item(),flush=True)
    torch.manual_seed(20908)
    exact=0
    for i in range(100):
        x.copy_(torch.randn_like(x[:46]).repeat(16,1))
        candidate();q=y.float().view(16,46,-1)
        exact+=int(torch.equal(q,q[:1].expand_as(q)))
    print('mutation_exact',exact,'/100',flush=True)
    graphs=[]
    for fn in (lambda:torch.mm(x,w.t(),out=y),candidate):
        fn();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):fn()
        graphs.append(g)
    samples={'A':[],'B':[]}
    for _ in range(5):
        for name,g in [('A',graphs[0]),('B',graphs[1]),('B',graphs[1]),('A',graphs[0])]:
            for _ in range(20):g.replay()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(100):g.replay()
            end.record();end.synchronize()
            samples[name].append(start.elapsed_time(end)*10)
    print('median_us',{k:statistics.median(v) for k,v in samples.items()},flush=True)


if __name__=='__main__':main()
