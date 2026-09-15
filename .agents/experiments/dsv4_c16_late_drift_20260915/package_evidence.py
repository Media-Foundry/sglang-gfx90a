"""Bounded diagnostic archive; retain full tensor manifest including local files."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
analysis=json.loads((root/'analysis.json').read_text())
assert analysis['first_differing_checkpoint']==24 and analysis['identical_all_captured_input_metadata']
files=[];tensor_manifest=[]
for arm in ('A','B'):
    assert json.loads((root/arm/f'P16-late-drift-{arm}.stop.json').read_text())['remaining']==[]
    for p in sorted((root/arm).rglob('*')):
        if not p.is_file():continue
        if p.suffix=='.pt':
            tensor_manifest.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bundled=p.name.startswith('rank-0-')))
            if not p.name.startswith('rank-0-'):continue
        elif p.suffix not in ('.json','.log','.sh','.patch'):continue
        assert p.stat().st_size<16*1024*1024
        files.append(p)
(root/'tensor-manifest.json').write_text(json.dumps(tensor_manifest,indent=2)+'\n')
for name in ('run.py','sweep.py','analyze.py','package_evidence.py','README.md','analysis.json','tensor-manifest.json'):
    files.append(root/name)
archive=root/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname=str(p.relative_to(root)))
assert archive.stat().st_size<90*1024*1024
manifest=dict(files=len(files),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k!='members'})
