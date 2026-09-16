"""Preserve pre-integration oracle and exact source/tensor-seed snapshots."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'oracle-v1-evidence.tar.gz';manifest=root/'oracle-v1-manifest.json'
assert not target.exists() and not manifest.exists()
record=json.loads((root/'oracle-v1.json').read_text())
assert record['status']=='complete' and len(record['cases'])==12
assert all(all(q['byte_exact'] for q in c['common_small_vs_large']) for c in record['cases'])
files=[Path(p) for p in record['sources']]+[root/'oracle-v1.json',root/'oracle-v1.log']
for name,digest in record['sources'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==digest
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),indent=2)+'\n')
print('TAIL ORACLE ARCHIVE',len(files),target.stat().st_size)
