"""Preserve the first rejected check: text [None] metadata excluded all calls."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent;repo=root.parents[2];d=root/'check'
assert not (d/'complete.json').exists()
assert not json.loads((d/'P8-first-table-check.stop.json').read_text())['remaining']
assert 'DSV4 first-mix table built:' not in (d/'P8-first-table-check.service.log').read_text()
target=root/'no-hit-evidence.tar.gz';assert not target.exists()
files={root/'service.py',root/'check-driver.log',root/'archive_no_hit.py'}
for name,digest in json.loads((d/'plan.json').read_text())['sources'].items():
    p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    files.add(p)
files.update(p for p in d.rglob('*') if p.is_file())
with tarfile.open(target,'w:gz') as archive:
    for p in sorted(files):archive.add(p,arcname=str(p.relative_to(repo)))
(root/'no-hit-manifest.json').write_text(json.dumps(dict(status='rejected_no_table_hits',
    archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),indent=2)+'\n')
print('ARCHIVED NO-HIT',target.stat().st_size)
