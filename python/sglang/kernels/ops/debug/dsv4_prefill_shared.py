"""Default-off row-invariant shared-expert GEMMs for native TP8 prefill."""
import logging
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

_logged = False

@triton.jit(do_not_specialize=["M"])
def _project(X, W, Y, M, N: tl.constexpr, K: tl.constexpr):
    m=tl.program_id(0)*128+tl.arange(0,128)
    n=tl.program_id(1)*128+tl.arange(0,128)
    k=tl.arange(0,128)
    acc=tl.zeros((128,128),tl.float32)
    for start in range(K//128):
        kk=start*128+k
        a=tl.load(X+m[:,None]*K+kk[None,:],m[:,None]<M,other=0)
        b=tl.load(W+n[None,:]*K+kk[:,None])
        acc=tl.dot(a,b,acc)
    tl.store(Y+m[:,None]*N+n[None,:],acc,m[:,None]<M)

def _linear(x,w):
    m,k=x.shape;n=w.shape[0]
    out=torch.empty((m,n),device=x.device,dtype=x.dtype)
    _project[(triton.cdiv(m,128),triton.cdiv(n,128))](
        x,w,out,m,n,k,num_warps=8,num_stages=2)
    return out

def shared(x, gate, down, limit):
    global _logged
    if (x.ndim!=2 or x.shape[1]!=4096 or not 8192<=x.shape[0]<=36864
            or gate.shape!=(512,4096) or down.shape!=(4096,256) or limit!=10.
            or any(t.dtype!=torch.bfloat16 or not t.is_cuda or
                   t.device!=x.device or not t.is_contiguous() for t in (x,gate,down))):
        raise ValueError("Stable shared requires TP8 BF16 M8192..36864/H4096/I256 and limit10")
    gu=_linear(x,gate)
    g,u=gu.chunk(2,dim=-1)
    # Match production's separate BF16 clamp, SiLU and multiply roundings.
    mid=F.silu(g.clamp(max=limit))*u.clamp(min=-limit,max=limit)
    out=_linear(mid,down)
    if not _logged:
        logging.getLogger(__name__).info("DSV4 diagnostic stable shared selected: M=%d I256",x.shape[0])
        _logged=True
    return out
