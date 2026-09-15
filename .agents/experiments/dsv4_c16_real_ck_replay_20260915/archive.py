import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert (root/'replay.json').exists() and (root/'reference.json').exists()
paths=[]
for arm in ('capture','capture-v2'):
    paths.extend(p for p in (root/arm).glob('*') if p.is_file() and p.suffix in ('.json','.log','.sh','.patch'))
paths.append(root/'capture-v2/fixture/manifest.json')
paths.extend(root.glob('*.log'))
paths.extend(root.glob('atomic-changed-*.pt'))
paths.extend([root/'selected-row-reference.pt',root/'replay.json',root/'reference.json'])
target=root/'evidence.tar.gz';assert not target.exists()
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(set(paths)):archive.add(path,arcname=str(path.relative_to(root)))
assert target.stat().st_size<90_000_000
info=dict(files=len(set(paths)),bytes=target.stat().st_size,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    fixture_local_only='capture-v2/fixture/*.pt; verified file/tensor hashes in fixture/manifest.json',
    failed_fixture_local_only='capture/fixture/*.pt; incomplete, never used for replay')
(root/'archive.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info))
