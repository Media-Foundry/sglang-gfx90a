"""Archive source/build provenance and completed oracle, excluding large binaries."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert json.loads((root/'build-v1/build.json').read_text())['status']=='complete'
screen=json.loads((root/'screen-v1.json').read_text())
assert screen['status']=='complete' and len(screen['cases'])==2
assert all(c['captured_stage1_exact'] and c['replay100_exact'] for c in screen['cases'])
target=root/'oracle-evidence.tar.gz';manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[p for p in sorted((root/'build-v1').rglob('*'))
       if p.is_file() and p.suffix in ('.json','.log','.hpp','.d')]
files += [root/'build-v1.log',root/'screen-v1.log']
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,
                sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('PRODUCER ORACLE ARCHIVE',len(files),target.stat().st_size)
