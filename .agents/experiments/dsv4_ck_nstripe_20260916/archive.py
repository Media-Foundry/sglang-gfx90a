"""Bounded prototype build/contract evidence, without binary/object bloat."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
result=json.loads((root/'screen.json').read_text())
assert result['status']=='complete' and len(result['cases'])==12
build_manifest=Path(json.loads((root/'build.json').read_text())['manifest'])
build=build_manifest.parent
target=root/'build-evidence.tar.gz';output=root/'archive-manifest.json'
assert not target.exists() and not output.exists()
files=[p for p in sorted(build.rglob('*')) if p.is_file() and p.suffix in ('.json','.log','.hpp','.d')]
fixtures=sorted({Path(c['fixture'])/'manifest.json' for c in result['cases']})
records=[]
with tarfile.open(target,'w:gz') as archive:
    for path in files:
        name=str(path.relative_to(root));archive.add(path,arcname=name)
        records.append(dict(path=str(path),member=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    for i,path in enumerate(fixtures):
        name=f'fixture-{i}-manifest.json';archive.add(path,arcname=name)
        records.append(dict(path=str(path),member=name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
output.write_text(json.dumps(dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,files=records),indent=2)+'\n')
print('BUILD ARCHIVE',len(records),target.stat().st_size)
