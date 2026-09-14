"""Package completed shared-expert causal/repair evidence, excluding tensors."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
run=root/'layer0-shared-wired'
assert (run/'complete.json').exists() and (run/'all-rank-summary.json').exists()
target=root/'shared-evidence.tar.gz';manifest=root/'shared-evidence-manifest.json'
assert not target.exists() and not manifest.exists()
inventory={}
for path in sorted(run.glob('trace-*/*.pt')):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk)
    inventory[str(path.relative_to(root))]=dict(bytes=path.stat().st_size,sha256=digest.hexdigest())
manifest.write_text(json.dumps(inventory,indent=2)+'\n')
files=[p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.sh','.log','.patch')]
files += [manifest]+[root/name for name in ('shared-oracle.json','shared-oracle.log',
    'shared-service-oracle.json','shared-service-oracle.log','shared-axes-oracle.json','shared-axes-oracle.log')]
for name in ('layer0-stable-qkv-wqb-wob-shared','layer0-shared-scope-probe'):
    files += [p for p in sorted((root/name).iterdir()) if p.is_file() and p.suffix in ('.json','.sh','.log','.patch')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
print(dict(files=len(files),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
