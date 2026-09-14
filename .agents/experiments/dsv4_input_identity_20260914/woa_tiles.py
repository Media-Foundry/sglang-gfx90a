"""Offline fixed-K-order wo_a tile screen, not a production selector."""
import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["M"])
def kernel(X, W, Y, M, N: tl.constexpr, K: tl.constexpr,
           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.zeros((BM,BN),tl.float32)
    for start in range(K//BK):
        kk=start*BK+k
        a=tl.load(X+m[:,None]*K+kk[None,:],m[:,None]<M,other=0)
        b=tl.load(W+n[None,:]*K+kk[:,None],n[None,:]<N,other=0)
        acc=tl.dot(a,b,acc)
    tl.store(Y+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


def project(x,w,tile):
    bm,bn,bk,warps=tile
    m,k=x.shape;n,wk=w.shape
    assert k==wk and k%bk==0 and x.is_contiguous() and w.is_contiguous()
    assert x.dtype==w.dtype==torch.bfloat16
    y=torch.empty((m,n),device=x.device,dtype=x.dtype)
    kernel[(triton.cdiv(m,bm),triton.cdiv(n,bn))](
        x,w,y,m,n,k,bm,bn,bk,num_warps=warps,num_stages=2)
    return y


TILES=[(64,128,64,4),(128,64,64,4),(64,128,32,4),
       (128,128,32,8),(128,128,64,8),(128,128,64,4)]
TILES2=[(32,128,128,4),(64,128,128,4),(128,128,128,8),
        (64,256,64,8),(64,256,128,8),(128,128,128,4)]
