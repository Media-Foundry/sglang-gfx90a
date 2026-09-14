"""Experimental exact C4 prefill query-row reuse; selection stays per row."""
import torch
import triton
import triton.language as tl

@triton.jit
def load_k(cache, page, cols, maxlen, SHUFFLE:tl.constexpr, DOT:tl.constexpr, FP8:tl.constexpr):
    d=tl.arange(0,128)
    pos=cols%64
    base=tl.maximum(page,0)*8448
    if SHUFFLE:
        offset=base+(pos[:,None]//SHUFFLE)*(SHUFFLE*128)+(d[None,:]//SHUFFLE)*(SHUFFLE*SHUFFLE)+(pos[:,None]%SHUFFLE)*SHUFFLE+d[None,:]%SHUFFLE
    else:
        offset=base+pos[:,None]*128+d[None,:]
    valid=cols<maxlen
    raw=tl.load(cache+offset,mask=valid[:,None],other=0)
    kv=raw.to(FP8,bitcast=True).to(DOT)
    sp=base+8192+pos*4
    b0=tl.load(cache+sp,mask=valid,other=0).to(tl.uint32)
    b1=tl.load(cache+sp+1,mask=valid,other=0).to(tl.uint32)
    b2=tl.load(cache+sp+2,mask=valid,other=0).to(tl.uint32)
    b3=tl.load(cache+sp+3,mask=valid,other=0).to(tl.uint32)
    scale=(b0|(b1<<8)|(b2<<16)|(b3<<24)).to(tl.float32,bitcast=True)
    return kv,scale


@triton.jit
def emit_runtime_m(q,w,out,row,cols,length,kv,scale,M,W:tl.constexpr,DOT:tl.constexpr,FP8:tl.constexpr):
    h=tl.arange(0,64)
    d=tl.arange(0,128)
    qv=tl.load(q+row*8192+h[:,None]*128+d[None,:],mask=row<M,other=0).to(FP8,bitcast=True).to(DOT)
    scores=tl.dot(kv,tl.trans(qv)).to(tl.float32)
    weights=tl.load(w+row*64+h,mask=row<M,other=0).to(tl.float32)
    reduced=tl.sum(tl.maximum(scores,0.)*weights[None,:],axis=1)
    result=tl.where((cols<length)&(length>512),reduced*scale,0.)
    tl.store(out+row*W+cols,result,mask=(row<M)&(cols<W))


@triton.jit(do_not_specialize=['M'])
def reuse_runtime_m(q,cache,w,lens,pages,out,M,W:tl.constexpr,NP:tl.constexpr,
          PT:tl.constexpr,BQ:tl.constexpr,BS:tl.constexpr,SHUFFLE:tl.constexpr,DOT:tl.constexpr,FP8:tl.constexpr):
    first=tl.program_id(0)*BQ
    start=tl.program_id(1)*BS
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
