"""Isolated multi-query K reuse; no production dispatch edits.

Reuse the existing empty-tile fixture and logical/physical Top-K checks, but
compare against the CURRENT nonempty control, not its superseded rectangular
dot implementation. Synthetic 8K causal requests, not a real-service capture.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

os.environ.setdefault('SGLANG_OPT_USE_TRITON_INDEXER_FULL','1')
os.environ.setdefault('SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER','3')
import torch
import triton
import triton.language as tl
from sglang.srt.layers.attention.dsv4 import indexer


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
          BQ:tl.constexpr,BS:tl.constexpr,SHUFFLE:tl.constexpr,DOT:tl.constexpr,FP8:tl.constexpr):
    first=tl.program_id(0)*BQ
    start=tl.program_id(1)*BS
    rows=first+tl.arange(0,BQ)
    lengths=tl.load(lens+rows,mask=rows<M,other=0)
    ids=tl.load(pages+rows*NP+start//64,mask=(rows<M)&(start//64<NP),other=-1)
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
                    page=tl.load(pages+row*NP+start//64)
                    kv,scale=load_k(cache,page,cols,length,SHUFFLE,DOT,FP8)
                    emit(q,w,out,row,cols,length,kv,scale,M,W,DOT,FP8)
                else:
                    tl.store(out+row*W+cols,0.,mask=(row<M)&(cols<W))
    else:
        tl.store(out+rows[:,None]*W+cols[None,:],0.,mask=(rows[:,None]<M)&(cols[None,:]<W))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bq',type=int,choices=(2,4),default=2)
    parser.add_argument('--runtime',action='store_true')
    parser.add_argument('--sizes',type=int,nargs='+',default=[8192,32768])
    parser.add_argument('--mutations',type=int,default=10)
    parser.add_argument('--replays',type=int,default=10)
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();assert not a.output.exists()
    assert torch.version.hip and 'gfx90a' in torch.cuda.get_device_properties(0).gcnArchName
    torch.manual_seed(20260915)
    root=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('empty_fixture',root/'scripts/rocm/check_dsv4_indexer_empty_tiles.py')
    fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)
    original=indexer.fp8_paged_mqa_logits_torch
    resources=[]
    def dispatch(q,kv,w,lengths,pages,schedule,width,clean_logits,*,skip_trivial_topk,skip_empty_tiles):
        assert skip_trivial_topk==512
        if not skip_empty_tiles:
            return original(q,kv,w,lengths,pages,schedule,width,clean_logits,
                            skip_trivial_topk=512,skip_empty_tiles=True)
        if a.runtime:
            from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
            assert a.bq==4
            out=prefill_query_reuse4(q,kv,w,lengths,pages,width,
                block_s=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S.get(),
                preshuffle_tile=(indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE
                    if indexer.aiter_can_use_preshuffle_paged_mqa() else 0),
                dot_fp16=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get(),
                fp8_fnuz=indexer.is_fp8_fnuz())
            assert out is not None,'Runtime unexpectedly declined fixture'
            return out
        m=q.shape[0]
        assert q.is_contiguous() and pages.is_contiguous() and w.is_contiguous()
        out=torch.empty(m,width,device=q.device,dtype=torch.float32)
        bs=indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_BLOCK_S.get()
        assert bs in (16,32,64)
        shuffle=indexer.INDEXER_K_CACHE_PRESHUFFLE_TILE if indexer.aiter_can_use_preshuffle_paged_mqa() else 0
        dot=tl.float16 if indexer.envs.SGLANG_DSV4_GFX90A_INDEXER_FP16_DOT.get() else tl.bfloat16
        fp8=tl.float8e4b8 if indexer.is_fp8_fnuz() else tl.float8e4nv
        obj=reuse[(triton.cdiv(m,a.bq),triton.cdiv(width,bs))](q.view(torch.uint8),kv.view(torch.uint8),w,lengths,pages,out,m,width,pages.shape[1],a.bq,bs,shuffle,dot,fp8,num_warps=4)
        if obj is not None and not resources:
            resources.append(dict(regs=obj.n_regs,spills=obj.n_spills,lds=obj.metadata.shared))
        return out
    indexer.fp8_paged_mqa_logits_torch=dispatch
    a.prefill=True;a.production=True;a.abba_cycles=3;a.timing_iterations=5
    result=dict(status='running',bq=a.bq,runtime=a.runtime,scope=__doc__,cases=[],
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                fp8_dtype=str(indexer.FP8_DTYPE),
                preshuffle=indexer.aiter_can_use_preshuffle_paged_mqa())
    for m in a.sizes:
        row=fixture.run_case(m,2048,2048,512,a)
        result['cases'].append(row);result['resources']=resources
        a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(row),flush=True)
    # Odd M, partial final key tile and unrelated per-row page tables exercise
    # fallback, unlike the aligned 8192-row per-request performance fixture.
    m,width,npages=17,577,30
    q=torch.randn(m,1,64,128,device='cuda').to(indexer.FP8_DTYPE)
    w=torch.randn(m,64,device='cuda')
    lens=torch.randint(0,width+1,(m,),device='cuda',dtype=torch.int32)
    table_width=triton.cdiv(width,64)
    pages=torch.randint(0,npages,(m,table_width+3),device='cuda',dtype=torch.int32)[:,:table_width]
    cache=torch.empty(npages,8448,device='cuda',dtype=torch.uint8)
    values=torch.randn(npages*64,128,device='cuda')
    loc=torch.arange(npages*64,device='cuda',dtype=torch.int32)
    fixture.triton_fused_store_indexer(values,cache,loc,64)
    def stage(candidate):
        scores=dispatch(q,cache.view(npages,64,1,132),w,lens,pages,None,width,False,
                        skip_trivial_topk=512,skip_empty_tiles=candidate)
        logical=torch.empty(m,512,device='cuda',dtype=torch.int32)
        physical=torch.empty_like(logical)
        fixture.topk_transform_512(scores,lens,pages,physical,64,logical)
        return scores,logical,physical
    graphs=[];outputs=[]
    for arm in (False,True):
        stage(arm)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):out=stage(arm)
        graphs.append(graph);outputs.append(out)
    for step in range(a.mutations):
        q.copy_(torch.randn(q.shape,device='cuda').to(indexer.FP8_DTYPE));w.normal_();values.normal_()
        fixture.triton_fused_store_indexer(values,cache,loc,64)
        pages.random_(0,npages);lens.random_(0,width+1)
        # Force active rows on different pages AND shared-page masked rows.
        lens[:a.bq]=width
        pages[:a.bq,0]=torch.arange(a.bq,device='cuda')
        if step%2==0:pages[a.bq:2*a.bq]=pages[a.bq]
        if step%3==0:q.zero_()  # cutoff ties
        for graph in graphs:graph.replay()
        assert torch.equal(outputs[0][0].view(torch.int32),outputs[1][0].view(torch.int32)),('ragged scores',step)
        assert torch.equal(outputs[0][1],outputs[1][1]),('ragged logical',step)
        assert torch.equal(outputs[0][2],outputs[1][2]),('ragged physical',step)
    result['ragged_mixed_page_mutations_exact']=a.mutations
    result['ragged_page_table_stride']=list(pages.stride())
    result['status']='complete';a.output.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
