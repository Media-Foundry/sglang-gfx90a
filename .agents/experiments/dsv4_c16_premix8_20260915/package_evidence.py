"""Package completed component checks, never claim pending service acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
full=json.loads((root/'full.json').read_text())
integrated=json.loads((root/'integrated.json').read_text())
assert full['status']==integrated['status']=='complete'
assert len(full['results'])==9 and len(integrated['results'])==4
assert sum(len(r['mutations']) for r in full['results'])==900
assert all(c['bits_exact'] and c['finite'] for r in full['results'] for c in r['mutations'])
assert all(r['bits_exact'] and r['mutations']==10 for r in integrated['results'])
files=[root/name for name in ('candidate.py','screen.py','integrated.py',
    'screen.json','screen.log','full.json','full.log','integrated.json','integrated.log')]
archive=root/'component-evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname=p.name)
manifest=dict(files=len(files),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),scope='Component only; service ABBA pending')
(root/'component-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
