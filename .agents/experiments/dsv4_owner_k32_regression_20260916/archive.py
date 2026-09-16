"""Preserve both completed and accepted shorter-length regression trials."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
target=root/'service-evidence.tar.gz'
manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[]
for length in ('8k','16k'):
    accepted=json.loads((root/('acceptance-'+length+'.json')).read_text())
    assert accepted['status']=='accepted_explicit_'+length+'_profile'
    for arm in ('A1','B','A2'):
        directory=root/(length+'-'+arm)
        files.extend(p for p in sorted(directory.rglob('*'))
                     if p.is_file() and p.suffix in ('.json','.log','.sh','.patch','.txt'))
        files.append(root/(length+'-'+arm+'-driver.log'))
files.append(root/'run.log')
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('REGRESSION ARCHIVE',len(files),target.stat().st_size)
