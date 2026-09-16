"""Preserve full eight-rank live-reference diagnostic, including owned cleanup."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
directory = root / 'check'
done = json.loads((directory / 'complete.json').read_text())
assert done['diagnostic'] and done['live_comparisons'] == 1376 and done['france_passed']
assert not json.loads((directory / 'P8-route-producer-check.stop.json').read_text())['remaining']
plan = json.loads((directory / 'plan.json').read_text())
assert all(hashlib.sha256((repo / p).read_bytes()).hexdigest() == h for p, h in plan['sources'].items())
target = root / 'check-evidence.tar.gz'
manifest = root / 'check-manifest.json'
assert not target.exists() and not manifest.exists()
files = sorted(p for p in directory.iterdir() if p.is_file())
with tarfile.open(target, 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(root)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in files]), indent=2) + '\n')
print('LIVE CHECK ARCHIVE', len(files), target.stat().st_size)
