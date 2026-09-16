"""Archive accepted regression evidence with per-file digests."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert json.loads((root/'acceptance.json').read_text())['status']=='passed_shorter_input_common_mhc_regression'
target=root/'service-evidence.tar.gz'
manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[]
for length in ('8k','16k'):
    for arm in ('A1','B','A2'):
        files.extend(p for p in sorted((root/(length+'-'+arm)).rglob('*'))
                     if p.is_file() and p.suffix in ('.json','.log','.sh','.patch','.txt'))
with tarfile.open(target,'w:gz') as archive:
    for p in files: archive.add(p,arcname=str(p.relative_to(root)))
record=dict(archive=target.name,bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('REGRESSION ARCHIVE',len(files),target.stat().st_size)
