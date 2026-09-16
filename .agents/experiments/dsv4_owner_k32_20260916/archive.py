"""Archive completed K32 service evidence only after explicit acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
acceptance=json.loads((root/'acceptance.json').read_text())
assert acceptance['status']=='accepted_explicit_32k_profile'
target=root/'service-evidence.tar.gz'
manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[]
for arm in ('A1','B','A2'):
    files.extend(p for p in sorted((root/arm).rglob('*'))
                 if p.is_file() and p.suffix in ('.json','.log','.sh','.patch','.txt'))
    files.append(root/(arm+'-driver.log'))
files.append(root/'run.log')
with tarfile.open(target,'w:gz') as archive:
    for path in files:
        archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
            bytes=target.stat().st_size,
            files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('SERVICE ARCHIVE',len(files),target.stat().st_size)
