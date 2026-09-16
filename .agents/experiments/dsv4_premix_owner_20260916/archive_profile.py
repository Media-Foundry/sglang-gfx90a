"""Archive the new profile separately; never overwrite scoring evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
directory=root/'profile'
assert json.loads((directory/'analysis.json').read_text())['snapshots']==128
target=root/'profile-evidence.tar.gz'
manifest=root/'profile-archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[p for p in sorted(directory.rglob('*')) if p.is_file() and p.suffix in ('.json','.sh','.log','.patch')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=[dict(path=str(p.relative_to(root)),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest(),bytes=p.stat().st_size) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('PROFILE ARCHIVE',len(files),record['bytes'],record['sha256'])
