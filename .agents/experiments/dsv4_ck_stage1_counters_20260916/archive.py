"""Archive small raw profiler artifacts, exact commands and source provenance."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert (root/'summary.json').exists()
target=root/'evidence.tar.gz';manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[p for p in sorted(root.rglob('*')) if p.is_file() and p.suffix in ('.json','.csv','.log')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('COUNTER ARCHIVE',len(files),target.stat().st_size)
