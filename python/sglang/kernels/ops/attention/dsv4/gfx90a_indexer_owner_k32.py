"""Default-off owner-only K32 logits, original-V4 TP8 eager-prefill consumer.

Query production, cache updates, Top-K and communication stay with the caller.
The caller must already have admitted the original-V4 native TP8 owner path.
"""
import torch
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

def eligible(local_rows, width, global_rows, *, preshuffle_tile, dot_fp16, fp8_fnuz):
    """Small/untested contracts retain the original K16 path."""
    return (
        global_rows is not None and 8192 <= global_rows <= 65536
        and 1024 <= local_rows <= ((global_rows + 127) // 128) * 16
        and local_rows % 16 == 0 and 2048 <= width <= 8192
        and preshuffle_tile in (0, 16) and not dot_fp16 and not fp8_fnuz
    )


def prefill_owner_k32(q, cache, weights, lengths, pages, width, *,
                      admitted_global_rows, preshuffle_tile, dot_fp16, fp8_fnuz):
    m = q.shape[0]
    if not eligible(m, width, admitted_global_rows, preshuffle_tile=preshuffle_tile,
                    dot_fp16=dot_fp16, fp8_fnuz=fp8_fnuz):
        return None
    if lengths.ndim == 2 and lengths.shape[-1] == 1:
        lengths = lengths.squeeze(-1)
    tensors = (q, cache, weights, lengths, pages)
    if not (
        q.shape == (m, 1, 64, 128) and q.dtype == torch.float8_e4m3fn
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
    ):
        return None
    out = torch.empty((m, width), dtype=torch.float32, device=q.device)
    reuse_grouped_grid[(triton.cdiv(m, 16) * triton.cdiv(width, 32),)](
        q.view(torch.uint8), cache.view(torch.uint8), weights, lengths, pages,
        out, m, width, pages.shape[1], pages.stride(0),
        16, 32, preshuffle_tile, tl.bfloat16, tl.float8e4nv, 1,
        num_warps=4, matrix_instr_nonkdim=16,
    )
    return out
