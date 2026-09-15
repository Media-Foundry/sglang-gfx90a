"""Bounded diagnostic evidence; preserve all remaining row hashes locally."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
target = root / 'evidence.tar.gz'
assert not target.exists()
paths = []
for arm in ('A', 'B'):
    for p in (root / arm).rglob('*'):
        if not p.is_file():
            continue
        if p.suffix in ('.json', '.log', '.sh', '.patch'):
            paths.append(p)
        elif p.suffix == '.pt' and (p.name == 'layer-0-rank-0-metadata.pt' or
                p.name.startswith(('layer-0-rank-6-', 'layer-1-rank-6-', 'layer-2-rank-6-'))):
            paths.append(p)
with tarfile.open(target, 'w:gz') as archive:
    for p in sorted(paths):
        archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
manifest = dict(archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    scope='All raw JSON/logs, canonical input metadata and selected frontier payloads; not all tensor files',
    files={str(p.relative_to(root)): dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)})
(root / 'evidence.json').write_text(json.dumps(manifest, indent=2) + '\n')
print(target, target.stat().st_size)
