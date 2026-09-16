"""Opt-in original-V4 TP8 eager-prefill logits ownership; query/KV stay full.

No persistent weights/workspace and no live-device-to-host planning. Only the
small row maps belong to one PagedIndexerMetadata/forward. Callers must enforce
model/mode/parallel/backend scope BEFORE adding a collective.
"""
import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.distributed as dist
import triton
import triton.language as tl

from .gfx90a_indexer_query_reuse import prefill_query_reuse4
from .topk import topk_transform_512


def host_plan(extend_lens, prefix_lens, rank):
    """Mirror metadata_kernel's seq_len//4; keep original query16 groups."""
    if not (0 <= rank < 8 and len(extend_lens) == len(prefix_lens)):
        raise ValueError('invalid ownership metadata')
    if not extend_lens or any(n <= 0 for n in extend_lens) or any(p < 0 for p in prefix_lens):
        raise ValueError('invalid prefix/extend length')
    lengths = np.concatenate([(np.arange(1,n+1,dtype=np.int32)+p)//4
                              for n,p in zip(extend_lens,prefix_lens,strict=True)])
    m=len(lengths)
    padded=np.pad(lengths,(0,(-m)%16))
    groups=np.flatnonzero(padded.reshape(-1,16).max(axis=1)>512)
    cap_groups=(len(groups)+7)//8
    owner_groups=np.full((8,cap_groups),-1,dtype=np.int64)
    for r in range(8):
        g=groups[r::8];owner_groups[r,:len(g)]=g
    rows=(owner_groups[:,:,None]*16+np.arange(16)).reshape(8,-1)
    valid=(np.repeat(owner_groups,16,axis=1)>=0)&(rows<m)
    rows=np.where(valid,rows,-1)
    inverse=np.full(m,-1,dtype=np.int32)
    flat=rows.reshape(-1)
    inverse[flat[flat>=0]]=np.flatnonzero(flat>=0).astype(np.int32)
    assert np.all(inverse[lengths>512]>=0)
    return lengths,np.maximum(rows[rank],0),valid[rank].astype(np.int32),inverse


@triton.jit
def reconstruct(gathered, inverse, lengths, pages, raw, physical,
                PT:tl.constexpr, WRITE_RAW:tl.constexpr):
    row=tl.program_id(0);k=tl.arange(0,512)
    length=tl.load(lengths+row)
    source=tl.load(inverse+row)
    if length>512:
        logical=tl.load(gathered+source*512+k)
    else:
        logical=tl.where(k<length,length-1-k,-1)
    page=tl.load(pages+row*PT+tl.maximum(logical,0)//64,mask=logical>=0,other=0)
    if WRITE_RAW:tl.store(raw+row*512+k,logical)
    tl.store(physical+row*512+k,tl.where(logical>=0,page*64+logical%64,-1))


@dataclass
class OwnerPlan:
    key:tuple
    rowids:torch.Tensor
    valid:torch.Tensor
    inverse:torch.Tensor
    expected:np.ndarray


def forward(*,q,cache,weights,lengths,pages,width,output,raw_output,
            batch,metadata,rank,group,preshuffle_tile,dot_fp16,fp8_fnuz):
    """Returns False only before any collective; errors after admission fail fast."""
    m=q.shape[0]
    wide = (os.getenv('SGLANG_DSV4_C4_PREFILL_QUERY_OWNER_WIDE', '0') == '1'
            and 8192 <= m <= 65536 and width > 2048)
    if not (q.shape==(m,1,64,128) and weights.shape==(m,64)
            and weights.dtype==torch.float32 and 512<=width<=(8192 if wide else 2048)
            and pages.shape[0]==m and pages.stride(1)==1
            and output.shape==(m,512) and output.is_contiguous()
            and (raw_output is None or raw_output.shape==(m,512) and raw_output.is_contiguous())
            and all(t.is_contiguous() for t in (q,cache,weights,lengths))):
        return False
    def cpu_list(value):
        if isinstance(value,torch.Tensor):
            if value.device.type!='cpu':return None
            value=value.tolist()
        return tuple(map(int,value)) if value is not None else None
    ext=cpu_list(batch.extend_seq_lens_cpu);pre=cpu_list(batch.extend_prefix_lens_cpu)
    if ext is None or pre is None or sum(ext)!=m or len(ext)!=len(pre):return False
    # A configured score-width cap must never turn longer live causal rows
    # into out-of-bounds Top-K reads. This is CPU metadata, not a device sync.
    if not ext or max((n+p)//4 for n,p in zip(ext,pre))>width:return False
    key=(ext,pre,rank,str(q.device))
    plan=getattr(metadata,'_gfx90a_owner_plan',None)
    if plan is None or plan.key!=key:
        expected,ids,valid,inverse=host_plan(ext,pre,rank)
        if len(ids)==0:return False
        plan=OwnerPlan(key,torch.from_numpy(ids).to(q.device),
                       torch.from_numpy(valid).to(q.device),
                       torch.from_numpy(inverse).to(q.device),expected)
        metadata._gfx90a_owner_plan=plan
    check=os.getenv('SGLANG_DSV4_DEBUG_PREFILL_OWNER_CHECK','0')=='1'
    lengths=lengths.reshape(-1)
    if check:
        torch.testing.assert_close(lengths,torch.from_numpy(plan.expected).to(q.device),atol=0,rtol=0)
    pq=q.view(torch.uint8).index_select(0,plan.rowids).view(q.dtype)
    pw=weights.index_select(0,plan.rowids)
    pl=lengths.index_select(0,plan.rowids)*plan.valid
    pp=pages.index_select(0,plan.rowids)
    kw=dict(block_s=16,preshuffle_tile=preshuffle_tile,dot_fp16=dot_fp16,
            fp8_fnuz=fp8_fnuz,query_group_size=16,runtime_m=True)
    owner_kw = dict(allow_wide=True, admitted_global_rows=m) if wide else {}
    scores=prefill_query_reuse4(pq,cache,pw,pl,pp,width,**kw,**owner_kw)
    assert scores is not None,'Admitted query-owner layout unsupported'
    local_phys=torch.empty((len(plan.rowids),512),device=q.device,dtype=torch.int32)
    local_raw=torch.empty_like(local_phys)
    topk_transform_512(scores,pl,pp,local_phys,64,local_raw)
    gathered=torch.empty((8*len(plan.rowids),512),device=q.device,dtype=torch.int32)
    # Exactly the integer RCCL API exercised by the eight-rank oracle. Do not
    # bitcast into a float custom-AG selector or add a separate peer allocator.
    dist.all_gather_into_tensor(gathered,local_raw,group=group)
    actual_raw=raw_output if raw_output is not None else torch.empty_like(output) if check else None
    reconstruct[(m,)](gathered,plan.inverse,lengths,pages,actual_raw,output,
                       pages.stride(0),actual_raw is not None,num_warps=4)
    if check:
        full_scores=prefill_query_reuse4(q,cache,weights,lengths,pages,width,**kw,
                                       **(dict(allow_wide=True) if wide else {}))
        ref_phys=torch.empty_like(output);ref_raw=torch.empty_like(output)
        topk_transform_512(full_scores,lengths,pages,ref_phys,64,ref_raw)
        torch.testing.assert_close(output,ref_phys,atol=0,rtol=0)
        torch.testing.assert_close(actual_raw,ref_raw,atol=0,rtol=0)
        a=full_scores.index_select(0,plan.rowids)[plan.valid.bool()]
        b=scores[plan.valid.bool()]
        assert torch.equal(a.view(torch.int32),b.view(torch.int32))
        if wide:
            print(f'[TP{rank}] wide-owner exact: rows={m} local_rows={len(plan.rowids)} '
                  f'width={width} scores=byte_exact logical=exact physical=exact',flush=True)
    if not getattr(metadata,'_gfx90a_owner_logged',False):
        print(f'[TP{rank}] prefill query-owner selected: rows={m} local_rows={len(plan.rowids)} '
              f'width={width} exchange_bytes={gathered.numel()*4} check={int(check)}',flush=True)
        metadata._gfx90a_owner_logged=True
    return True
