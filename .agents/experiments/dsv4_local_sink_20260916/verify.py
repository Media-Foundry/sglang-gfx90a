"""Export compact auditable sink values and preserve corrected-baseline evidence."""
import hashlib
import json
from pathlib import Path
import tarfile
import torch
from safetensors import safe_open

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'check'
target=root/'summary.json';assert not target.exists()
done=json.loads((out/'complete.json').read_text());assert done['france_passed']
assert done['input_echo_exact']==32 and all(c['checkpoint_slice_exact'] for c in done['sink_checks'])
plan=json.loads((out/'plan.json').read_text())
assert plan['sources']=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in plan['sources']}
model=Path('/home/pc/models/modelscope');key='layers.20.attn.attn_sink'
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
with safe_open(model/index[key],framework='pt',device='cpu') as f:full=f.get_tensor(key)
values=[]
for rank in range(8):
    meta=json.loads((out/'capture'/f'rank-{rank}.json').read_text())
    data=torch.load(out/'capture'/f'rank-{rank}.pt',weights_only=True,map_location='cpu',mmap=True)
    sink=data['attn_sink'];expected=full[rank*8:(rank+1)*8]
    assert torch.equal(sink,expected)
    assert hashlib.sha256(sink.view(torch.uint8).numpy().tobytes()).hexdigest()==meta['hashes']['attn_sink']
    values.append(dict(rank=rank,observed=sink.tolist(),expected=expected.tolist()))
timing=json.loads((out/'corrected-baseline.json').read_text())
assert len(timing['rounds'])==3 and all(r['input_echo_exact'] and r['cached_tokens']==[0]*16 for r in timing['rounds'])
assert not json.loads((out/'P16-local-sink.stop.json').read_text())['remaining']
result=dict(scope='Original V4 TP8 C16x8K native AR; corrected sink; H16 disabled',
    input_tok_s=timing['median_input_tok_s'],rates=[r['aggregate_input_tok_s'] for r in timing['rounds']],
    wave_times=[r['prefill_wall_s'] for r in timing['rounds']],kv_tokens=1048576,
    france_passed=True,quality_repeat_exact=done['quality_repeat_exact'],quality_requests=16,
    observed_sink_values=values,all_model_drift_fixed=False,
    performance_scope='Three warm waves in one new process; not a correction-vs-old ABBA')
target.write_text(json.dumps(result,indent=2)+'\n')
archivepath=root/'evidence.tar.gz';assert not archivepath.exists()
files=set(root.glob('*.py'))|set(root.glob('*.json'))
for ext in ('*.json','*.log','*.sh'):files.update(out.glob(ext))
files.update((out/'capture').glob('*.json'))
with tarfile.open(archivepath,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name in plan['sources']:archive.add(repo/name,arcname='sources/'+name)
manifest=dict(archive=archivepath.name,sha256=hashlib.sha256(archivepath.read_bytes()).hexdigest(),
    bytes=archivepath.stat().st_size,full_tensor_fixtures_included=False)
(root/'archive-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='observed_sink_values'},indent=2))
