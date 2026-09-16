"""Isolated CTA-order experiment; Q/K math imported from measured runtime-M kernel."""
import triton
import triton.language as tl
from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import load_k, emit_runtime_m

@triton.jit(do_not_specialize=['M'])
def reuse_grouped_grid(q,cache,w,lens,pages,out,M,W:tl.constexpr,NP:tl.constexpr,
          PT:tl.constexpr,BQ:tl.constexpr,BS:tl.constexpr,SHUFFLE:tl.constexpr,DOT:tl.constexpr,FP8:tl.constexpr,GQ:tl.constexpr):
    pid=tl.program_id(0)
    nk=tl.cdiv(W,BS)
    qgroup=(pid//(GQ*nk))*GQ+pid%GQ
    ktile=(pid%(GQ*nk))//GQ
    first=qgroup*BQ
    start=ktile*BS
    rows=first+tl.arange(0,BQ)
    lengths=tl.load(lens+rows,mask=rows<M,other=0)
    ids=tl.load(pages+rows*PT+start//64,mask=(rows<M)&(start//64<NP),other=-1)
    active=(rows<M)&(lengths>512)&(lengths>start)
    maxlen=tl.max(tl.where(active,lengths,0),axis=0)
    cols=start+tl.arange(0,BS)
    if maxlen>start:
        minpage=tl.min(tl.where(active,ids,2147483647),axis=0)
        maxpage=tl.max(tl.where(active,ids,-1),axis=0)
        if minpage==maxpage:
            kv,scale=load_k(cache,minpage,cols,maxlen,SHUFFLE,DOT,FP8)
            for i in tl.static_range(BQ):
                row=first+i
                length=tl.load(lens+row,mask=row<M,other=0)
                emit_runtime_m(q,w,out,row,cols,length,kv,scale,M,W,DOT,FP8)
        else:
            # Correct fallback for unrelated page tables/request boundaries.
            for i in tl.static_range(BQ):
                row=first+i
                length=tl.load(lens+row,mask=row<M,other=0)
                if row<M and length>512 and length>start:
                    page=tl.load(pages+row*PT+start//64)
                    kv,scale=load_k(cache,page,cols,length,SHUFFLE,DOT,FP8)
                    emit_runtime_m(q,w,out,row,cols,length,kv,scale,M,W,DOT,FP8)
                else:
                    tl.store(out+row*W+cols,0.,mask=(row<M)&(cols<W))
    else:
        tl.store(out+rows[:,None]*W+cols[None,:],0.,mask=(rows[:,None]<M)&(cols[None,:]<W))
