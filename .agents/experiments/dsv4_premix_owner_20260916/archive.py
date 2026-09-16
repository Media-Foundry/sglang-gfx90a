"""Package bounded reproducibility evidence; retain all originals locally."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
archive=root/'service-evidence.tar.gz'
manifest=root/'archive-manifest.json'
assert not archive.exists() and not manifest.exists()
assert json.loads((root/'summary.json').read_text())['status']=='complete'
files=[]
for arm in ('audit','check','A1','B','A2'):
    for path in sorted((root/arm).rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log','.sh','.patch'):
            files.append(path)
with tarfile.open(archive,'w:gz') as tar:
    for path in files:
        tar.add(path,arcname=str(path.relative_to(root)))
records=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
              sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
payload=dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
             bytes=archive.stat().st_size,files=records)
manifest.write_text(json.dumps(payload,indent=2)+'\n')
print('ARCHIVED',len(files),'files',payload['bytes'],'bytes',payload['sha256'])
