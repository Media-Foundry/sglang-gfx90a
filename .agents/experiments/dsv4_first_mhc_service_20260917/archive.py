"""Freeze the complete uninstrumented first-boundary ABBA and source versions."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
assert json.loads((root/'summary.json').read_text())['status']=='complete'
files={root/n for n in ('service.py','run.py','analyze.py','archive.py','summary.json','quality-review.json','run.log')}
for arm in ('A1','B','A2'):
    d=root/arm
    assert not json.loads((d/f'P8-first-mhc-{arm}.stop.json').read_text())['remaining']
    for name,digest in json.loads((d/'plan.json').read_text())['sources'].items():
        p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        files.add(p)
    files.update(p for p in d.rglob('*') if p.is_file())
    files.add(root/(arm+'-driver.log'))
records=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
with tarfile.open(target,'w:gz') as archive:
    for p in sorted(files):archive.add(p,arcname=str(p.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,files=records,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest()),indent=2)+'\n')
print('ARCHIVED',len(files),target.stat().st_size)
