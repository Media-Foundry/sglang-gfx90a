"""Archive one completed service trial and its bounded component evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
run=root/'layer0-stable-qkv-wqb'
assert (run/'complete.json').exists() and (run/'all-rank-summary.json').exists()
target=root/'wqb-evidence.tar.gz';manifest=root/'wqb-evidence-manifest.json'
assert not target.exists() and not manifest.exists()
inventory={}
for path in sorted(run.glob('trace-*/*.pt')):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    inventory[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
manifest.write_text(json.dumps(inventory,indent=2)+'\n')
files=[x for x in sorted(run.iterdir()) if x.is_file() and x.suffix in ('.json','.sh','.log','.patch')]
files += [manifest] + [root/name for name in (
    'projection-library-screen.json','projection-library-screen.log',
    'projection-library-screen-retry.log','wqb-service-oracle.json','wqb-service-oracle.log')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
print(dict(bytes=target.stat().st_size,files=len(files),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
