"""Independent audit of completed diagnostic, fixing the harness's 43/41-layer count."""
import collections
import hashlib
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'check'
target=out/'validated.json';assert not target.exists()
plan=json.loads((out/'plan.json').read_text())
for p,digest in plan['sources'].items():
    source=out/'service-at-run.py' if p.endswith('/dsv4_h16_service_20260916/service.py') else repo/p
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest,p
config=json.loads(Path('/home/pc/models/modelscope/config.json').read_text())
layers=[i for i,r in enumerate(config['compress_ratios'][:config['num_hidden_layers']]) if r in (4,128)]
assert layers==list(range(2,43))
logpath=out/'P16-h16-check.service.log';logs=logpath.read_text()
assert 'Scheduler hit an exception' not in logs and 'H16 peer output mismatch' not in logs
records=re.findall(r'\[TP(\d+)\] H16 peer output exact: layer=(\d+) rows=(\d+)',logs)
counts=collections.Counter((int(r),int(l),int(m)) for r,l,m in records)
expected={(r,l,m):n for r in range(8) for l in layers for m,n in ((32767,1),(32768,2),(32766,1))}
assert dict(counts)==expected
audits=re.findall(r'\[TP(\d+)\] H16 peer logical audit: layer=(\d+) rows=(\d+) hash=([0-9a-f]+)',logs)
assert len(audits)==8*len(layers)
mapping={(int(r),int(l)):h for r,l,m,h in audits}
for l in layers:
    for r in range(0,8,2):assert mapping[r,l]==mapping[r+1,l]
info=json.loads((out/'P16-h16-check.server-info.json').read_text())
assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
data=json.loads((out/'check.json').read_text())
assert len(data['rounds'])==1
assert data['rounds'][0]['cached_tokens']==[0]*16 and data['rounds'][0]['completion_lengths']==[1]*16
assert 'paris' in json.loads((out/'France.json').read_text())['text'].lower()
assert not json.loads((out/'P16-h16-check.stop.json').read_text())['remaining']
gpus=json.loads((out/'P16-h16-check-after-stop.gpu.json').read_text())
assert not any(isinstance(p.get('process_info'),dict) for g in gpus for p in g.get('process_list',[]))
result=dict(all_eligible_layers_exact=True,layers=layers,exact_comparisons=len(records),
    logical_audits=len(audits),france_passed=True,kv_tokens=1048576,
    correction='Original harness wrongly expected pure-SWA layers0/1 to enter C4/C128-only candidate; no runtime change.',
    log_sha256=hashlib.sha256(logpath.read_bytes()).hexdigest())
target.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
