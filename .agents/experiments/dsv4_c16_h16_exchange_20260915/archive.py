"""Bounded evidence archive; full model tensor fixtures remain local."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
target = root / 'capture-evidence.tar.gz'
assert not target.exists()
paths = []
for name in ('capture', 'capture-v2'):
    for path in (root / name).rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.log', '.sh', '.patch'):
            paths.append(path)
manifest = {str(p.relative_to(root)): {'bytes': p.stat().st_size,
             'sha256': hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(paths)}
with tarfile.open(target, 'w:gz') as archive:
    for path in sorted(paths):
        archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
record = {'archive_sha256': hashlib.sha256(target.read_bytes()).hexdigest(),
          'files': manifest, 'tensor_fixtures_included': False}
(root / 'capture-evidence.json').write_text(json.dumps(record, indent=2) + '\n')
print(target, target.stat().st_size)
