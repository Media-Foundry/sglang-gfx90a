"""Archive raw service evidence without checkpoint tensors or compiled binaries."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
archive=root/'service-evidence.tar.gz';manifest=root/'archive-manifest.json'
assert not archive.exists() and not manifest.exists()
assert json.loads((root/'summary.json').read_text())['status']=='measured-pending-quality-review'
files=[]
for arm in ['check','A1','B','A2']:
    directory=root/arm
    assert (directory/'complete.json').exists()
    files.extend(p for p in directory.iterdir() if p.is_file() and p.suffix in ('.json','.log','.sh','.patch'))
with tarfile.open(archive,'w:gz') as out:
    for path in sorted(files):out.add(path,arcname=str(path.relative_to(root)))
manifest.write_text(json.dumps(dict(archive=archive.name,sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}),indent=2)+'\n')
print(archive,archive.stat().st_size)
