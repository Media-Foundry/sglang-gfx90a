"""Archive actual small-tail policy component verification and source snapshots."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'integrated-evidence.tar.gz';manifest=root/'integrated-manifest.json'
assert not target.exists() and not manifest.exists()
record=json.loads((root/'integrated-v1.json').read_text())
assert record['status']=='complete' and len(record['cases'])==21
assert all(c['selector_exact'] and all(q['selector_exact'] for q in c['checks']) for c in record['cases'])
assert all(all(q['byte_exact'] for q in c['common_small_vs_large']) for c in record['cases'])
files=[Path(p) for p in record['sources']]+[root/'integrated-v1.json',root/'integrated-v1.log',
    repo/'python/sglang/srt/model_executor/runner/eager_runner.py',
    repo/'test/registered/unit/layers/test_dsv4_prefill_tail_policy.py']
for name,digest in record['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),indent=2)+'\n')
print('INTEGRATED TAIL ARCHIVE',len(files),target.stat().st_size)
