"""Default-off, synchronized one-layer oracle; normal query outputs unchanged."""
import hashlib
import json
import os
from pathlib import Path
import statistics

import torch
import torch.distributed as dist

_done=False


def diagnose(*,indexer,batch,x,q_lora,positions,q,weights,cache,lengths,pages,width,
             preshuffle_tile,dot_fp16,fp8_fnuz):
    global _done
    directory=os.getenv('SGLANG_DSV4_DEBUG_OWNER_PRODUCER_DIR')
    if not directory or _done or indexer.layer_id!=20:return
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    if not mix_pair_active():return
    from sglang.srt.distributed.parallel_state import get_tp_group
    tp=get_tp_group();rank=tp.rank_in_group
    assert tp.world_size==8 and batch.forward_mode.name=='EXTEND'
    assert not indexer.use_fp4_indexer and not torch.cuda.is_current_stream_capturing()
    _done=True
    torch.cuda.synchronize();dist.barrier(group=tp.cpu_group)
    if rank==0:
        _run(Path(directory),indexer,batch,x,q_lora,positions,q,weights,cache,
             lengths,pages,width,preshuffle_tile,dot_fp16,fp8_fnuz)
    dist.barrier(group=tp.cpu_group)


def _run(root,indexer,batch,x,q_lora,positions,q,weights,cache,lengths,pages,width,
         preshuffle_tile,dot_fp16,fp8_fnuz):
    from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_owner import host_plan
    from sglang.kernels.ops.attention.dsv4.gfx90a_indexer_query_reuse import prefill_query_reuse4
    from sglang.kernels.ops.attention.dsv4.topk import topk_transform_512
    root.mkdir(parents=True,exist_ok=False)
    m=len(x);ext=list(map(int,batch.extend_seq_lens_cpu));pre=list(map(int,batch.extend_prefix_lens_cpu))
    assert m==sum(ext) and q.shape==(m,1,64,128)
    def digest(t):return hashlib.sha256(t.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
    def delta(a,b):
        assert a.shape==b.shape and a.dtype==b.dtype
        bits=(a.contiguous().view(torch.uint8)!=b.contiguous().view(torch.uint8))
        xf,yf=a.float(),b.float()
        return dict(changed_bytes=int(bits.sum()),max_abs=float((xf-yf).abs().max()),
            relative_l2=float(torch.linalg.vector_norm(xf-yf)/torch.linalg.vector_norm(xf).clamp_min(1e-30)),
            finite=bool(torch.isfinite(xf).all() and torch.isfinite(yf).all()))
    def produce(xx,ql,pos):
        initial=indexer.compute_weights(xx,skip_scale=True)
        qq,ww=indexer.compute_q(ql,pos,initial)
        return qq.unsqueeze(1),ww.squeeze(2)
    def measure(fn):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(5):fn()
        b.record();b.synchronize();return a.elapsed_time(b)/5
    report=dict(diagnostic_only=True,layer=20,rank=0,rows=m,extend_lens=ext,prefix_lens=pre,
        input_sha256=digest(batch.input_ids),position_sha256=digest(positions),
        x_sha256=digest(x),q_lora_sha256=digest(q_lora),original_q_sha256=digest(q),
        weights_sha256=digest(weights),owners=[],
        wq_b_quant_method=type(indexer.wq_b.quant_method).__name__,
        weights_quant_method=type(indexer.weights_proj.quant_method).__name__,
        weight_scale=float(indexer.weight_scale))
    parameters={}
    for name,module in [('wq_b',indexer.wq_b),('weights_proj',indexer.weights_proj)]:
        parameters[name]={key:dict(shape=list(value.shape),dtype=str(value.dtype),sha256=digest(value))
                          for key,value in module.named_parameters()}
    report['parameters']=parameters
    # Keep complete real producer inputs locally; parameter contracts are hashed.
    path=root/'inputs.pt'
    torch.save(dict(x=x.detach().cpu(),q_lora=q_lora.detach().cpu(),positions=positions.detach().cpu(),
                    input_ids=batch.input_ids.detach().cpu()),path)
    report['inputs_file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    qfull,wfull=produce(x,q_lora,positions)
    report['full_recompute']=dict(q=delta(q,qfull),weights=delta(weights,wfull))
    assert not report['full_recompute']['q']['changed_bytes'] and not report['full_recompute']['weights']['changed_bytes'],report['full_recompute']
    kw=dict(block_s=16,preshuffle_tile=preshuffle_tile,dot_fp16=dot_fp16,
            fp8_fnuz=fp8_fnuz,query_group_size=16,runtime_m=True)
    lengths=lengths.reshape(-1)
    for owner in range(8):
        expected,rowids,valid,_=host_plan(ext,pre,owner)
        ids=torch.from_numpy(rowids).to(x.device);mask=torch.from_numpy(valid).to(x.device).bool()
        pl=lengths.index_select(0,ids)*mask.int();pp=pages.index_select(0,ids)
        xs=x.index_select(0,ids);ql=q_lora.index_select(0,ids);pos=positions.index_select(0,ids)
        qa,wa=q.index_select(0,ids),weights.index_select(0,ids)
        raw_full=indexer.wq_b(q_lora)[0].index_select(0,ids)
        raw_part=indexer.wq_b(ql)[0]
        qb,wb=produce(xs,ql,pos)
        sa=prefill_query_reuse4(qa,cache,wa,pl,pp,width,**kw)
        sb=prefill_query_reuse4(qb,cache,wb,pl,pp,width,**kw)
        assert sa is not None and sb is not None
        pa=torch.empty((len(ids),512),device=x.device,dtype=torch.int32);pb=torch.empty_like(pa)
        ra=torch.empty_like(pa);rb=torch.empty_like(pa)
        topk_transform_512(sa,pl,pp,pa,64,ra);topk_transform_512(sb,pl,pp,pb,64,rb)
        item=dict(owner=owner,rows=len(ids),valid_rows=int(mask.sum()),
            raw_projection=delta(raw_full[mask],raw_part[mask]),
            q=delta(qa[mask],qb[mask]),weights=delta(wa[mask],wb[mask]),
            scores=delta(sa[mask],sb[mask]),logical_id_changed=int((ra[mask]!=rb[mask]).sum()),
            physical_id_changed=int((pa[mask]!=pb[mask]).sum()))
        qr,wr=produce(xs,ql,pos)
        item['repeat']=dict(q=delta(qb[mask],qr[mask]),weights=delta(wb[mask],wr[mask]))
        report['owners'].append(item)
        (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
        print('OWNER PRODUCER CHECK',owner,item['q']['changed_bytes'],item['logical_id_changed'],flush=True)
    # Complete producer comparison includes live row gathers on the candidate.
    def full_call():return produce(x,q_lora,positions)
    def owner_call():return produce(x.index_select(0,ids),q_lora.index_select(0,ids),positions.index_select(0,ids))
    for _ in range(3):full_call();owner_call()
    samples={'A':[],'B':[]}
    for _ in range(3):
        for label,fn in [('A',full_call),('B',owner_call),('B',owner_call),('A',full_call)]:
            samples[label].append(measure(fn))
    report['producer_samples_ms']=samples
    report['producer_median_ms']={key:statistics.median(values) for key,values in samples.items()}
    report['status']='complete'
    (root/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print('OWNER PRODUCER ORACLE COMPLETE',report['producer_median_ms'],flush=True)
