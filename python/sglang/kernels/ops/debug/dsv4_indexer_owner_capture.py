"""Default-off, first-forward cross-rank indexer input audit; never a benchmark."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import torch

_seen = set()

def selected_layers():
    value=os.getenv('SGLANG_DSV4_DEBUG_INDEXER_OWNER_LAYERS','2,20,42')
    layers=tuple(int(x.strip()) for x in value.split(','))
    assert layers and len(layers)==len(set(layers))
    assert all(2<=x<=42 and x%2==0 for x in layers),layers
    return layers

def digest(raw):
    raw=np.ascontiguousarray(raw)
    return hashlib.sha256(memoryview(raw).cast('B') if raw.size else b'').hexdigest()

def logical_rows(pages, count, tile):
    """Undo runtime K preshuffle, retaining exact FP8 codes and FP32 scale bytes."""
    assert pages.dtype == np.uint8 and pages.ndim == 2 and pages.shape[1] == 8448
    assert 0 <= count <= len(pages)*64 and tile in (0, 8, 16, 32, 64)
    k = pages[:, :8192]
    if tile:
        # Runtime address: row_group*(tile*128) + col_group*tile**2
        #                 + row_in_tile*tile + col_in_tile.
        k = k.reshape(-1,64//tile,128//tile,tile,tile).transpose(0,1,3,2,4)
    k = np.ascontiguousarray(k).reshape(-1,128)[:count]
    scales = np.ascontiguousarray(pages[:,8192:]).reshape(-1,4)[:count]
    return k,scales

def capture(*, layer_id, rank, batch, x, q_lora, q, weights, positions,
            seq_lens, page_table, cache, preshuffle_tile, dot_fp16, fp8_fnuz):
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    directory=os.getenv('SGLANG_DSV4_DEBUG_INDEXER_OWNER_DIR')
    if not directory or not mix_pair_active() or layer_id not in selected_layers(): return
    if (rank,layer_id) in _seen: return
    if torch.cuda.is_current_stream_capturing(): return
    assert q.ndim==4 and q.shape[1:]==(1,64,128)
    m=q.shape[0]
    assert 8192<=m<=65536 and x.shape[0]==q_lora.shape[0]==m
    assert batch.forward_mode.name=='EXTEND' and rank in range(8)
    lengths=list(map(int,batch.extend_seq_lens_cpu))
    prefixes=list(map(int,batch.extend_prefix_lens_cpu))
    assert sum(lengths)==m and len(lengths)==len(prefixes)
    path=Path(directory);path.mkdir(parents=True,exist_ok=True)
    target=path/f'rank-{rank}-layer-{layer_id}.json'
    assert not target.exists()
    ids=batch.input_ids[:m].detach().cpu()
    pos=positions.detach().cpu()
    lens=seq_lens.reshape(-1).detach().cpu().numpy()
    table=page_table.detach().cpu().numpy()
    assert len(lens)==len(table)==len(pos)==m
    bounds=np.cumsum([0,*lengths])
    representatives=[int(end)-1 for end in bounds[1:]]
    per_request=[]
    for row in representatives:
        count=int(lens[row]);assert count>=0
        page_ids=table[row,:((count+63)//64)]
        assert len(page_ids)==(count+63)//64
        per_request.append((count,page_ids))
    used=np.unique(np.concatenate([pages for _,pages in per_request])).astype(np.int64)
    assert len(used)>0 and int(used.min())>=0 and int(used.max())<cache.shape[0]
    raw_cache=cache.view(torch.uint8).reshape(cache.shape[0],8448)
    packed=raw_cache.index_select(0,torch.as_tensor(used,device=cache.device)).cpu()
    mapping=np.searchsorted(used,table)
    found=(mapping<len(used)) & (used[np.minimum(mapping,len(used)-1)]==table)
    needed=np.arange(table.shape[1])[None,:] < ((lens[:,None]+63)//64)
    assert bool(np.all(found | ~needed)), 'Referenced page omitted from compact snapshot'
    remapped=np.where(found,mapping,0).astype(np.int32)
    logical=[]
    for request,(count,pages) in enumerate(per_request):
        raw=packed.numpy()[np.searchsorted(used,pages)]
        keys,scales=logical_rows(raw,count,preshuffle_tile)
        logical.append(dict(request=request,rows=count,keys_sha256=digest(keys),
                            scale_bytes_sha256=digest(scales)))
    samples=sorted(set([0,m-1,*representatives,*[min(m-1,int(b)+2048) for b in bounds[:-1]]]))
    record=dict(rank=rank,layer=layer_id,rows=m,extend_lens=lengths,prefix_lens=prefixes,
        input_ids_sha256=digest(ids.numpy()),positions_sha256=digest(pos.numpy()),
        seq_lens_sha256=digest(lens),sample_rows=samples,logical_kv=logical,
        used_physical_pages=used.tolist(),
        preshuffle_tile=preshuffle_tile,dot_fp16=bool(dot_fp16),fp8_fnuz=bool(fp8_fnuz),
        tensors={},diagnostic_only=True,logical_decoder_version=2)
    packed_path=path/f'rank-{rank}-layer-{layer_id}-pages.pt'
    assert not packed_path.exists()
    torch.save(dict(pages=packed,used_physical_pages=used.tolist(),
                    requests=[dict(count=count,page_ids=page_ids.tolist())
                              for count,page_ids in per_request]),packed_path)
    record['packed_file']=packed_path.name
    record['packed_sha256']=hashlib.sha256(packed_path.read_bytes()).hexdigest()
    saved=dict(input_ids=ids,positions=pos,seq_lens=torch.from_numpy(lens.copy()),
               sample_rows=samples,samples={})
    full = {} if rank==0 and layer_id==20 and os.getenv('SGLANG_DSV4_DEBUG_INDEXER_OWNER_FULL','1')=='1' else None
    for name,tensor in (('x',x),('q_lora',q_lora),('q',q),('weights',weights)):
        raw=tensor.detach().contiguous().view(torch.uint8).cpu()
        record['tensors'][name]=dict(shape=list(tensor.shape),dtype=str(tensor.dtype),
            original_stride=list(tensor.stride()),sha256=digest(raw.numpy()))
        saved['samples'][name]=raw[samples].clone()
        if full is not None and name in ('q','weights'): full[name]=raw
    sample_path=path/f'rank-{rank}-layer-{layer_id}.pt'
    assert not sample_path.exists();torch.save(saved,sample_path)
    record['sample_file']=sample_path.name
    record['sample_sha256']=hashlib.sha256(sample_path.read_bytes()).hexdigest()
    if full is not None:
        full.update(input_ids=ids,positions=pos,seq_lens=torch.from_numpy(lens.copy()),
                    page_table=torch.from_numpy(remapped),cache=packed.reshape(-1,64,1,132),
                    metadata=record)
        full_path=path/'layer20-rank0-full.pt'
        assert not full_path.exists();torch.save(full,full_path)
        record['full_file']=full_path.name
        record['full_sha256']=hashlib.sha256(full_path.read_bytes()).hexdigest()
    target.write_text(json.dumps(record,indent=2)+'\n')
    _seen.add((rank,layer_id))
    print(f'[TP{rank}] indexer owner audit: layer={layer_id} rows={m} logical_requests={len(lengths)}',flush=True)
