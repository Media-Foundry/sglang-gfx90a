"""Default-off one-layer peer-attention oracle capture; never a speed probe."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

_seen=set()


def canonical_slots(indices):
    """Relabel physical slots by first occurrence; preserve duplicates/sentinels."""
    indices=np.asarray(indices,dtype=np.int64)
    valid=indices>=0
    used,first,inverse=np.unique(indices[valid],return_index=True,return_inverse=True)
    order=np.argsort(first)
    labels=np.empty(len(used),dtype=np.int32)
    labels[order]=np.arange(len(used),dtype=np.int32)
    result=np.full(indices.shape,-1,dtype=np.int32)
    result[valid]=labels[inverse]
    return used[order],result


def capture(layer_id,rank,batch,args,output):
    directory=os.getenv('SGLANG_DSV4_DEBUG_ATTN_PEER_CAPTURE_DIR')
    if not directory or layer_id!=20 or rank in _seen:return
    assert batch.forward_mode.name=='EXTEND' and not torch.cuda.is_current_stream_capturing()
    q=args['q'];m=len(q)
    assert q.shape==(m,8,512) and q.dtype==torch.bfloat16 and 8192<=m<=65536
    root=Path(directory);root.mkdir(parents=True,exist_ok=True)
    target=root/f'rank-{rank}.json';assert not target.exists()
    compact={};checked=dict(args)
    for bank,ids in (('unified_kv','kv_indices_prefix'),('kv_extend','kv_indices_extend')):
        raw=args[ids].detach().cpu().numpy()
        used,canonical=canonical_slots(raw)
        if len(used):
            assert int(used.max())<len(args[bank])
            values=args[bank].index_select(0,torch.from_numpy(used).to(q.device))
        else:
            values=torch.zeros((1,512),device=q.device,dtype=q.dtype)
        checked[bank]=values;checked[ids]=torch.from_numpy(canonical).to(q.device)
        compact[bank]=values.detach().cpu();compact[ids]=torch.from_numpy(canonical)
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import runtime
    reference=runtime.prefill(**checked,num_stages=1)
    assert torch.equal(output.view(torch.uint8),reference.view(torch.uint8)), 'KV canonical relabel changed output'
    for name in ('q','kv_indptr_prefix','kv_indptr_extend'):
        compact[name]=args[name].detach().contiguous().cpu()
    compact['attn_sink']=args['attn_sink'][:8].detach().contiguous().cpu()
    compact['output']=output.detach().contiguous().cpu()
    compact['input_ids']=batch.input_ids.detach().cpu()
    compact['positions']=batch.positions.detach().cpu()
    hashes={name:hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for name,t in compact.items()}
    record=dict(layer=layer_id,rank=rank,rows=m,
        extend_lens=list(map(int,batch.extend_seq_lens_cpu)),
        prefix_lens=list(map(int,batch.extend_prefix_lens_cpu)),
        original_q_stride=list(q.stride()),original_sink_shape=list(args['attn_sink'].shape),
        softmax_scale=float(args['softmax_scale']),canonical_output_byte_exact=True,
        hashes=hashes,shapes={name:list(t.shape) for name,t in compact.items()})
    assert sum(record['extend_lens'])==m
    path=root/f'rank-{rank}.pt';assert not path.exists();torch.save(compact,path)
    record['file_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    target.write_text(json.dumps(record,indent=2)+'\n');_seen.add(rank)
    print(f'[TP{rank}] peer attention fixture captured: layer={layer_id} rows={m} canonical_exact=1',flush=True)
