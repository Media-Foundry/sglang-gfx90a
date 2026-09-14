"""Preserve detailed service markers and positive/negative component screens."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
files=[root/'markers-B-detailed-run.log']
for pattern in ('*.json','*.log','*.sh','*.patch','markers/*.json'):
    files+=sorted((root/'markers-B-detailed').glob(pattern))
files+=sorted((root/'marker-smoke-details').glob('*.json'))
files+=[root/'marker-smoke-details.log']
for stem in ('fp32-mhc-screen','post-fused4-screen','post-fused4-screen-v2','post-fused4-full'):
    files += [root/f'{stem}.json',root/f'{stem}.log']
assert len(files)==len(set(files))
assert all(p.is_file() and p.stat().st_size<16*1024*1024 for p in files)
archive=root/'detailed-marker-evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for path in files:out.add(path,arcname=str(path.relative_to(root)))
manifest=dict(archive=archive.name,bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(root/'detailed-marker-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}),'files',len(files))
