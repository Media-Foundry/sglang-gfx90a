"""Archive CK diagnostic metadata and small changed-row tensors, not weights."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
target = root / 'evidence.tar.gz'
assert not target.exists()
paths = [p for p in (root / 'capture').rglob('*') if p.is_file()
         and p.suffix in ('.json', '.sh', '.log', '.patch')]
paths.extend(root.glob('atomic-changed-*.pt'))
if (root / 'selected-row-reference.pt').exists():
    paths.append(root / 'selected-row-reference.pt')
with tarfile.open(target, 'w:gz') as archive:
    for p in sorted(paths):
        archive.add(p, arcname=str(p.relative_to(root)), recursive=False)
(root / 'evidence.json').write_text(json.dumps(dict(
    archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    full_fixture_included=False,
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)}), indent=2)+'\n')
print(target, target.stat().st_size)
