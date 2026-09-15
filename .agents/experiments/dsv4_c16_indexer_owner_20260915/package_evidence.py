"""Archive bounded audit evidence; never publish the 270MiB full-Q fixture."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert json.loads((root/'owner-rccl-v2.json').read_text())['status']=='complete'
assert json.loads((root/'analysis-v2.json').read_text())['all_checked_replicated_values_exact']
for folder in ('capture','capture-v2'):
    assert json.loads((root/folder/'P16-owner-audit.stop.json').read_text())['remaining']==[]
files=[]
for folder in ('capture','capture-v2','oracle-sources'):
    for p in sorted((root/folder).rglob('*')):
        if not p.is_file() or p.name=='layer20-rank0-full.pt':continue
        if p.suffix not in ('.json','.pt','.log','.py','.cuh','.sh','.patch'):continue
        assert p.stat().st_size<16*1024*1024,p
        files.append(p)
for name in ('capture.py','analyze.py','compare_captures.py','owner_oracle.py',
             'test_plan.py','package_evidence.py','README.md','analysis.json',
             'analysis-v2.json','cross-process.json','owner-rccl.json','owner-rccl-v2.json'):
    files.append(root/name)
archive=root/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p in files:out.add(p,arcname=str(p.relative_to(root)))
assert archive.stat().st_size<90*1024*1024
manifest=dict(files=len(files),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k!='members'})
