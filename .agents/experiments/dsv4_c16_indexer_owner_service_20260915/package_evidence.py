"""Archive completed service ABBA with source, input and lifecycle evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
assert summary['formal_input_echoes']==192
assert (root/'quality-review.json').exists() and (root/'manual-review.md').exists()
files=[]
for arm in ('check','A1','B','A2'):
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-owner-{arm}.stop.json').read_text())['remaining']==[]
    for p in sorted((root/arm).rglob('*')):
        if p.is_file() and p.suffix in ('.json','.log','.sh','.patch'):
            assert p.stat().st_size<16*1024*1024
            files.append(p)
for name in ('run.py','sweep.py','analyze.py','review.py','accept.py','package_evidence.py',
             'README.md','manual-review.md','summary.json','quality-review.json',
             'acceptance.json','tested_helper.py','final-gpu.json'):
    files.append(root/name)
archive=root/'service-evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname=str(p.relative_to(root)))
manifest=dict(files=len(files),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k!='members'})
