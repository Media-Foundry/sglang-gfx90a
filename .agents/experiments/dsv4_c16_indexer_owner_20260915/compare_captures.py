"""Cross-process input identity and sampled drift; no float-hash inference."""
import json
from pathlib import Path
import torch

root=Path(__file__).resolve().parent
out=root/'cross-process.json';assert not out.exists()
result=dict(scope='Same first-forward rows across two fresh accepted-path processes; no query-owner service path.',layers={})
for layer in (2,20,42):
    paths=[root/d/'data'/f'rank-0-layer-{layer}.json' for d in ('capture','capture-v2')]
    records=[json.loads(p.read_text()) for p in paths]
    for key in ('rows','extend_lens','prefix_lens','input_ids_sha256','positions_sha256','seq_lens_sha256','sample_rows'):
        assert records[0][key]==records[1][key],key
    samples=[torch.load(p.parent/r['sample_file'],weights_only=True)['samples'] for p,r in zip(paths,records)]
    tensors={}
    for name in ('x','q_lora','q','weights'):
        info=records[0]['tensors'][name]
        dtype=getattr(torch,info['dtype'].split('.')[-1])
        a,b=[s[name].view(dtype).float() for s in samples]
        assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
        diff=b-a
        tensors[name]=dict(full_hash_exact=info['sha256']==records[1]['tensors'][name]['sha256'],
            sampled_max_abs=float(diff.abs().max()),sampled_relative_l2=float(diff.norm()/a.norm().clamp_min(1e-30)),
            sampled_mismatch_elements=int((a!=b).sum()),sampled_elements=a.numel())
    result['layers'][str(layer)]=dict(identical_input_metadata=True,tensors=tensors)
responses=[json.loads((root/d/'responses.json').read_text()) for d in ('capture','capture-v2')]
by_id=[{r['meta_info']['id']:r for r in rs} for rs in responses]
assert set(by_id[0])==set(by_id[1])
result['first_completion_text_exact']=sum(by_id[0][rid]['text']==by_id[1][rid]['text'] for rid in by_id[0])
result['requests']=len(by_id[0])
result['limitation']='Sample metrics are not full-tensor error bounds. V1 logical-KV hashes are superseded and NOT compared. No first-divergent operator is identified.'
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
