"""Bounded source/log/JSON evidence; no model tensors, DSOs or object files."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
assert (root/'quality-summary.json').exists() and (root/'sink-audit.json').exists()
files=set(root.glob('*.py'))|set(root.glob('*.json'))|set(root.glob('*.cu'))
for arm in ('check','A1','B','A2','quality-A1','quality-B'):
    directory=root/arm
    for ext in ('*.json','*.log','*.sh','*.py'):files.update(directory.glob(ext))
sources=json.loads((root/'A1/plan.json').read_text())['sources']
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name,digest in sources.items():
        path=repo/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        archive.add(path,arcname='sources/'+name)
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,evidence_files=len(files),source_files=len(sources),
    excludes='Full fixtures, model weights, DSO/object files and compiler cache')
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
