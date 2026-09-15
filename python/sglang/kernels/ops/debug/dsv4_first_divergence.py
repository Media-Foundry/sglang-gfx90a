"""Default-off first-forward full-state/row-hash audit. Never a speed probe."""
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

_batch = None
_callbacks = {}


def fingerprint(raw, rows=None):
    raw = np.ascontiguousarray(raw)
    flat = memoryview(raw).cast('B') if raw.size else memoryview(b'')
    whole = hashlib.sha256(flat).hexdigest()
    if rows is None:
        return whole, None
    assert rows > 0 and raw.nbytes % rows == 0
    stride = raw.nbytes // rows
    # Full SHA proves whole-tensor identity. Independent 128-bit row hashes
    # locate changes without treating a tiny selected sample as comprehensive.
    hashes = b''.join(hashlib.blake2b(flat[i*stride:(i+1)*stride],digest_size=16).digest()
                      for i in range(rows))
    return whole, np.frombuffer(hashes,dtype=np.uint8).copy().reshape(rows,16)


def get_probe(layer, batch, positions):
    global _batch
    directory = os.getenv('SGLANG_DSV4_DEBUG_FIRST_DIV_DIR')
    if not directory:
        return None
    from sglang.srt.layers.dsv4_prefill_experiments import mix_pair_active
    if not mix_pair_active() or not 20 <= layer <= 24:
        return None
    if torch.cuda.is_current_stream_capturing():
        return None
    assert batch.forward_mode.name == 'EXTEND'
    if _batch is None:
        _batch = batch  # Strong identity prevents Python object-id reuse.
    if batch is not _batch:
        return None
    from sglang.srt.distributed import get_tp_group
    rank = get_tp_group().rank_in_group
    key = (layer,rank)
    if key in _callbacks:
        return _callbacks[key]
    assert not os.getenv('SGLANG_DSV4_DEBUG_STAGE_DUMP_DIR')
    assert not os.getenv('SGLANG_DSV4_DEBUG_ATTN_DUMP_DIR')
    root = Path(directory);root.mkdir(parents=True,exist_ok=True)
    pos=positions.detach().cpu();ids=batch.input_ids.detach().cpu()
    n=positions.numel()
    ext=list(map(int,batch.extend_seq_lens_cpu));pre=list(map(int,batch.extend_prefix_lens_cpu))
    assert len(ids)==n==sum(ext) and 8192<=n<=65536
    bounds=np.cumsum([0,*ext])
    # Samples are only supplemental numeric values. ALL rows get row hashes.
    wanted=set(range(2216,2401,8))|set(range(2230,2247))
    chosen=[i for i in range(int(bounds[-2]),n) if int(pos[i]) in wanted]
    chosen=sorted(set([0,n-1,*[int(x) for x in bounds[:-1]],*chosen]))
    stem=f'layer-{layer}-rank-{rank}'
    meta=dict(layer=layer,rank=rank,rows=n,extend_lens=ext,prefix_lens=pre,
              input_sha256=fingerprint(ids.numpy())[0],position_sha256=fingerprint(pos.numpy())[0],
              sampled_rows=chosen,diagnostic_only=True)
    meta_path=root/(stem+'-metadata.pt');assert not meta_path.exists()
    torch.save(dict(metadata=meta,input_ids=ids,positions=pos),meta_path)
    seen=set();counter=0

    def dump(name,value,**_):
        nonlocal counter
        if name in seen:return
        seen.add(name);counter+=1
        target=root/(stem+'-'+name+'.json');assert not target.exists()
        record=dict(layer=layer,rank=rank,stage=name,sequence=counter,metadata_file=meta_path.name)
        if value is None:
            record['is_none']=True
        else:
            assert isinstance(value,torch.Tensor),type(value)
            cpu=value.detach().contiguous().view(torch.uint8).cpu()
            row_tensor=value.ndim>0 and value.shape[0]==n and name not in ('input_ids','positions')
            digest,hashes=fingerprint(cpu.numpy(),n if row_tensor else None)
            record.update(shape=list(value.shape),dtype=str(value.dtype),sha256=digest,
                          row_tensor=row_tensor,is_none=False)
            if row_tensor:
                payload=dict(row_hashes=torch.from_numpy(hashes),samples=cpu[chosen].clone())
                path=root/(stem+'-'+name+'.pt');assert not path.exists()
                torch.save(payload,path)
                record.update(values_file=path.name,file_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        target.write_text(json.dumps(record,indent=2)+'\n')
        if name=='ffn_out':print(f'[TP{rank}] first-div layer completed: layer={layer} rows={n}',flush=True)
    _callbacks[key]=dump
    return dump
