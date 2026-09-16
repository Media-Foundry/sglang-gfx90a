"""Archive completed ABBA independently of whether its gain passes acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root / 'service-evidence.tar.gz'
manifest = root / 'archive-manifest.json'
assert not target.exists() and not manifest.exists()
assert json.loads((root / 'summary.json').read_text())['status'] == 'complete'
files = []
for arm in ('A1', 'B', 'A2'):
    directory = root / arm
    assert (directory / 'complete.json').is_file()
    assert not json.loads((directory / f'P8-route-producer-{arm}.stop.json').read_text())['remaining']
    plan = json.loads((directory / 'plan.json').read_text())
    assert all(hashlib.sha256((repo / p).read_bytes()).hexdigest() == h for p,h in plan['sources'].items())
    files.extend(p for p in directory.iterdir() if p.is_file())
files.extend(root / p for p in ('service.py', 'run.py', 'analyze.py', 'accept.py',
                               'test_accept.py', 'summary.json', 'run.log'))
for name in ('acceptance.json', 'acceptance.log', 'A1-driver.log', 'B-driver.log', 'A2-driver.log'):
    path = root / name
    if path.is_file(): files.append(path)
with tarfile.open(target, 'w:gz') as archive:
    for path in sorted(files):
        archive.add(path, arcname=str(path.relative_to(root)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(files)]), indent=2) + '\n')
print('SERVICE ARCHIVE', len(files), target.stat().st_size)
