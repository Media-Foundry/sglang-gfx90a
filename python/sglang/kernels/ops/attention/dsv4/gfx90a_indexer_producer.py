"""Opt-in eager-prefill Q ownership. Full weights projection and cache retained.

Caller enforces original-V4 TP8 native EXTEND scope. All fallback decisions
precede query/cache work and collectives; admitted contract failures are fatal.
Metadata always retains full-M rows, while Q has explicit owner-row mapping.
"""
import os
import threading

import torch
import torch.distributed as dist

from .gfx90a_indexer_owner import OwnerPlan, host_plan, reconstruct
from .gfx90a_indexer_query_reuse import prefill_query_reuse4
from .topk import topk_transform_512


def cpu_plan_inputs(ext, pre, rows, width):
    def convert(value):
        if isinstance(value, torch.Tensor):
            if value.device.type != 'cpu': return None
            value = value.tolist()
        return tuple(map(int, value)) if value is not None else None
    ext, pre = convert(ext), convert(pre)
    if (ext is None or pre is None or not ext or len(ext) != len(pre)
            or sum(ext) != rows or not 8192 <= rows <= 65536
            or not 512 <= width <= 2048 or min(ext) <= 0 or min(pre) < 0
            or max((n+p)//4 for n,p in zip(ext,pre)) > width):
        return None
    return ext, pre


def forward(*, backend, indexer, batch, metadata, x, q_lora, positions, lengths,
            pages, width, output, raw_output, cache, update_cache, rank, group,
            preshuffle_tile, dot_fp16, fp8_fnuz, mark=None):
    m = len(x)
    cpu = cpu_plan_inputs(batch.extend_seq_lens_cpu, batch.extend_prefix_lens_cpu, m, width)
    if cpu is None: return False
    if not (x.shape == (m,4096) and q_lora.shape == (m,1024)
            and positions.shape == (m,) and positions.is_contiguous()
            and x.dtype == q_lora.dtype == torch.bfloat16
            and indexer.wq_b.weight.shape == (8192,1024)
            and indexer.wq_b.weight.dtype == torch.bfloat16
            and indexer.wq_b.weight.is_contiguous() and indexer.wq_b.bias is None
            and getattr(indexer.wq_b,'_use_cached_block_fp8_bf16_weight',False)
            and lengths.numel() == m and lengths.is_contiguous()
            and pages.shape[0] == m and pages.stride(1) == 1
            and output.shape == (m,512) and output.is_contiguous()
            and (raw_output is None or raw_output.shape == (m,512) and raw_output.is_contiguous())
            and cache.is_contiguous() and cache.shape[1:] == (64,1,132)
            and all(t.device == x.device for t in (q_lora,positions,lengths,pages,output,cache))):
        return False
    ext,pre = cpu
    key = (ext,pre,rank,str(x.device))
    plan = getattr(metadata,'_gfx90a_owner_plan',None)
    if plan is None or plan.key != key:
        expected,ids,valid,inverse = host_plan(ext,pre,rank)
        if not len(ids): return False
        plan = OwnerPlan(key,torch.from_numpy(ids).to(x.device),
                         torch.from_numpy(valid).to(x.device),
                         torch.from_numpy(inverse).to(x.device),expected)
        metadata._gfx90a_owner_plan = plan
    # One handle per runner/backend + host thread + live stream, not per layer.
    # Retaining stream objects prevents accidental reuse of a destroyed stream ID.
    stream = torch.cuda.current_stream(x.device)
    handles = getattr(backend,'_gfx90a_owner_producer_handles',None)
    if handles is None:
        handles = {}; backend._gfx90a_owner_producer_handles = handles
    handle_key = (threading.get_ident(),x.device.index,stream.cuda_stream)
    if handle_key not in handles:
        from .gfx90a_rocblas_linear import Linear
        handles[handle_key] = (stream,Linear())
    linear = handles[handle_key][1]
    from sglang.kernels.ops.attention.dsv4 import fused_q_indexer_rope_hadamard_quant
    if mark is not None: mark(49,f'indexer_query_rows_{m}',absolute=True)
    precomputed = getattr(batch,'_dsv4_precomputed_index_weights',None)
    initial = (precomputed.pop(indexer.layer_id)
               if precomputed is not None and indexer.layer_id in precomputed
               else indexer.compute_weights(x,skip_scale=True))
    assert initial.shape == (m,64)
    if mark is not None: mark(50,'indexer_weights_done',absolute=True)
    def query(ql,pos,iw):
        raw = linear(ql,indexer.wq_b.weight).view(-1,64,128)
        q,w = fused_q_indexer_rope_hadamard_quant(raw,iw,indexer.weight_scale,indexer.freqs_cis,pos)
        return q.unsqueeze(1),w.squeeze(2)
    pq,pw = query(q_lora.index_select(0,plan.rowids),positions.index_select(0,plan.rowids),
                  initial.index_select(0,plan.rowids))
    if mark is not None: mark(51,'indexer_query_done',absolute=True)
    update_cache()  # Full x, never owner-only; later queries need every index key.
    if mark is not None: mark(52,'indexer_compressor_done',absolute=True)
    lengths = lengths.reshape(-1)
    pl = lengths.index_select(0,plan.rowids)*plan.valid
    pp = pages.index_select(0,plan.rowids)
    kw = dict(block_s=16,preshuffle_tile=preshuffle_tile,dot_fp16=dot_fp16,
              fp8_fnuz=fp8_fnuz,query_group_size=16,runtime_m=True)
    scores = prefill_query_reuse4(pq,cache,pw,pl,pp,width,**kw)
    assert scores is not None,'Admitted producer layout unsupported'
    local_phys = torch.empty((len(plan.rowids),512),device=x.device,dtype=torch.int32)
    local_raw = torch.empty_like(local_phys)
    topk_transform_512(scores,pl,pp,local_phys,64,local_raw)
    gathered = torch.empty((8*len(plan.rowids),512),device=x.device,dtype=torch.int32)
    dist.all_gather_into_tensor(gathered,local_raw,group=group)
    check = os.getenv('SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK','0') == '1'
    actual_raw = raw_output if raw_output is not None else torch.empty_like(output) if check else None
    reconstruct[(m,)](gathered,plan.inverse,lengths,pages,actual_raw,output,
                       pages.stride(0),actual_raw is not None,num_warps=4)
    if check:
        torch.testing.assert_close(lengths,torch.from_numpy(plan.expected).to(x.device),atol=0,rtol=0)
        qfull,wfull = query(q_lora.contiguous(),positions,initial)
        mask = plan.valid.bool()
        assert torch.equal(qfull.view(torch.uint8).index_select(0,plan.rowids)[mask],pq.view(torch.uint8)[mask])
        assert torch.equal(wfull.index_select(0,plan.rowids)[mask],pw[mask])
        full_scores = prefill_query_reuse4(qfull,cache,wfull,lengths,pages,width,**kw)
        assert torch.equal(full_scores.index_select(0,plan.rowids)[mask],scores[mask])
        ref_phys = torch.empty_like(output);ref_raw = torch.empty_like(output)
        topk_transform_512(full_scores,lengths,pages,ref_phys,64,ref_raw)
        torch.testing.assert_close(output,ref_phys,atol=0,rtol=0)
        torch.testing.assert_close(actual_raw,ref_raw,atol=0,rtol=0)
    if not getattr(metadata,'_gfx90a_owner_producer_logged',False):
        print(f'[TP{rank}] prefill query-producer selected: rows={m} local_rows={len(plan.rowids)} '
              f'width={width} check={int(check)} full_weights=1 rocblas=explicit',flush=True)
        metadata._gfx90a_owner_producer_logged = True
    if mark is not None:
        mark(53,'indexer_owner_chain_done',absolute=True)
        mark(54,'indexer_topk_done',absolute=True)
    return True
