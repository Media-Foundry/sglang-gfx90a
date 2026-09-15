"""CPU-only compare captured consumed sink entries with checkpoint TP head ownership."""
import hashlib
import json
from pathlib import Path
import torch
from safetensors import safe_open

root=Path(__file__).resolve().parent;target=root/'sink-audit.json';assert not target.exists()
model=Path('/home/pc/models/modelscope')
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
key='layers.20.attn.attn_sink';shard=model/index[key]
with safe_open(shard,framework='pt',device='cpu') as f:expected=f.get_tensor(key)
assert expected.shape==(64,)
fixture=root.parent/'dsv4_c16_h16_exchange_20260915/capture-v2/fixture'
records=[]
for rank in range(8):
    meta=json.loads((fixture/f'rank-{rank}.json').read_text())
    data=torch.load(fixture/f'rank-{rank}.pt',map_location='cpu',weights_only=True,mmap=True)
    sink=data['attn_sink'];assert sink.shape==(8,)
    digest=hashlib.sha256(sink.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    assert digest==meta['hashes']['attn_sink']
    wanted=expected[rank*8:(rank+1)*8]
    records.append(dict(rank=rank,consumed_equals_checkpoint_first8=torch.equal(sink,expected[:8]),
        consumed_equals_own_head_shard=torch.equal(sink,wanted),
        max_abs_wrong_slice=float((sink-wanted).abs().max()),consumed_sha256=digest))
assert all(r['consumed_equals_checkpoint_first8'] for r in records)
assert [r['consumed_equals_own_head_shard'] for r in records]==[True]+[False]*7
result=dict(layer=20,key=key,checkpoint_shard=str(shard),records=records,
    finding='Captured TP1..7 consume rank0 sink values. Unified model call passes self.attn_sink rather than existing _local_attn_sink result.',
    causal_scope='Deterministic head-ownership error; not proof of the cause of cross-run token drift.')
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
