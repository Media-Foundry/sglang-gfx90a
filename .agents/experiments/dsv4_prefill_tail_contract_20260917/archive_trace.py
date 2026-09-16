"""Archive completed row-aligned probes and their exact source versions."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'trace-evidence.tar.gz';manifest=root/'trace-manifest.json'
assert not target.exists() and not manifest.exists()
assert json.loads((root/'trace-analysis.json').read_text())['status']=='complete'
files={root/'analyze_trace.py',root/'trace-analysis.json',root/'archive_trace.py'}
for arm in ('mixed','serial'):
    d=root/f'trace-{arm}'
    assert not json.loads((d/f'P32k-tail-policy-trace-{arm}.stop.json').read_text())['remaining']
    for name,digest in json.loads((d/'plan.json').read_text())['sources'].items():
        p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        files.add(p)
    files.update(p for p in d.rglob('*') if p.is_file())
    files.add(root/f'trace-{arm}-driver.log')
files=sorted(files)
records=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(target,'w:gz',compresslevel=1) as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,files=records,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scored_performance=False),indent=2)+'\n')
print('TRACE ARCHIVED',len(files),target.stat().st_size)
