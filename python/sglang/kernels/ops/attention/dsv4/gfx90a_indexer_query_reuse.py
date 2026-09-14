"""Experimental exact C4 prefill query-row reuse; selection stays per row."""
import torch
import triton
import triton.language as tl

_compile_shapes_seen = set()

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
def emit(q,w,out,row,cols,length,kv,scale,M:tl.constexpr,W:tl.constexpr,DOT:tl.constexpr,FP8:tl.constexpr):
    h=tl.arange(0,64)
    d=tl.arange(0,128)
    qv=tl.load(q+row*8192+h[:,None]*128+d[None,:],mask=row<M,other=0).to(FP8,bitcast=True).to(DOT)
    scores=tl.dot(kv,tl.trans(qv)).to(tl.float32)
    weights=tl.load(w+row*64+h,mask=row<M,other=0).to(tl.float32)
    reduced=tl.sum(tl.maximum(scores,0.)*weights[None,:],axis=1)
    result=tl.where((cols<length)&(length>512),reduced*scale,0.)
    tl.store(out+row*W+cols,result,mask=(row<M)&(cols<W))


@triton.jit
def reuse(q,cache,w,lens,pages,out,M:tl.constexpr,W:tl.constexpr,NP:tl.constexpr,
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
                emit(q,w,out,row,cols,length,kv,scale,M,W,DOT,FP8)
        else:
            # Correct fallback for unrelated page tables/request boundaries.
            for i in tl.static_range(BQ):
                row=first+i
                length=tl.load(lens+row,mask=row<M,other=0)
                if row<M and length>512 and length>start:
                    page=tl.load(pages+row*PT+start//64)
                    kv,scale=load_k(cache,page,cols,length,SHUFFLE,DOT,FP8)
                    emit(q,w,out,row,cols,length,kv,scale,M,W,DOT,FP8)
                else:
                    tl.store(out+row*W+cols,0.,mask=(row<M)&(cols<W))
    else:
        tl.store(out+rows[:,None]*W+cols[None,:],0.,mask=(rows[:,None]<M)&(cols[None,:]<W))


def prefill_query_reuse4(q, cache, weights, lengths, pages, width, *,
                        block_s, preshuffle_tile, dot_fp16, fp8_fnuz,
                        query_group_size=4, runtime_m=False, trace_rank=None,
                        allow_wide=False):
    """Return None for unsupported contracts; never repack query/cache tensors."""
    m = q.shape[0]
    max_width = 8192 if (
        allow_wide and query_group_size == 16 and runtime_m and 8192 <= m <= 65536
    ) else 2048
    dtype = torch.float8_e4m3fnuz if fp8_fnuz else torch.float8_e4m3fn
    if lengths.ndim == 2 and lengths.shape[-1] == 1:
        lengths = lengths.squeeze(-1)
    tensors = (q, cache, weights, lengths, pages)
    if not (
        m > 0 and q.shape == (m, 1, 64, 128) and q.dtype == dtype
        and cache.ndim == 4 and cache.shape[1:] == (64, 1, 132)
        and cache.element_size() == 1
        and weights.shape == (m, 64) and weights.dtype == torch.float32
        and lengths.shape == (m,) and lengths.dtype == torch.int32
        and pages.ndim == 2 and pages.shape[0] == m
        and pages.shape[1] >= triton.cdiv(width, 64)
        and pages.dtype == torch.int32 and pages.stride(1) == 1
        and all(t.is_contiguous() for t in tensors[:-1])
        and all(t.device == q.device for t in tensors)
        and q.device.type == "cuda" and bool(torch.version.hip)
        and "gfx90a" in torch.cuda.get_device_properties(q.device).gcnArchName
        and 512 <= width <= max_width and block_s == 16
        and preshuffle_tile in (0, 8, 16)
        and query_group_size in (4, 8, 16)
    ):
        return None
    out = torch.empty((m, width), dtype=torch.float32, device=q.device)
    kernel = reuse
    if runtime_m and query_group_size == 16:
        from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_runtime_m import reuse_runtime_m

        kernel = reuse_runtime_m
    compiled = kernel[(triton.cdiv(m, query_group_size), triton.cdiv(width, block_s))](
        q.view(torch.uint8), cache.view(torch.uint8), weights, lengths, pages,
        out, m, width, pages.shape[1], pages.stride(0), query_group_size, block_s, preshuffle_tile,
        tl.float16 if dot_fp16 else tl.bfloat16,
        tl.float8e4b8 if fp8_fnuz else tl.float8e4nv, num_warps=4,
    )
    if trace_rank is not None and compiled is not None:
        alignment=tuple(t.data_ptr()%16 for t in tensors)
        key=(trace_rank,m,width,pages.shape[1],pages.stride(0),query_group_size,runtime_m,alignment)
        if key not in _compile_shapes_seen:
            print(f"[TP{trace_rank}] DSV4 indexer compile-shape rows={m} width={width} "
                  f"pages={pages.shape[1]} page_stride={pages.stride(0)} group={query_group_size} "
                  f"runtime_m={int(runtime_m and query_group_size == 16)} "
                  f"align={','.join(map(str,alignment))} "
                  f"artifact={compiled.hash} object={id(compiled)}",flush=True)
            _compile_shapes_seen.add(key)
    return out
