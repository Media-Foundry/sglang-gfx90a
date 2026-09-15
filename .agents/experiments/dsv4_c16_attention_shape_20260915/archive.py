import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
for name in ('screen.json','reload.json'):
    assert json.loads((root/name).read_text())['status']=='complete'
assert (root/'overlap.json').exists()
paths=[p for p in root.iterdir() if p.suffix in ('.py','.json','.log','.md','.pt')]
archive=root/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for path in sorted(paths):out.add(path,arcname=path.name)
assert archive.stat().st_size<90_000_000
record=dict(files=len(paths),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(name=p.name,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)])
(root/'archive.json').write_text(json.dumps(record,indent=2)+'\n')
print({k:v for k,v in record.items() if k!='members'})
