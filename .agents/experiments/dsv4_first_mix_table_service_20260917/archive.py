"""Freeze complete table ABBA, failed-attempt provenance, and runtime sources."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
archive = root/'evidence.tar.gz'
assert not archive.exists()
sources = set()
files = set()
for arm in ('check', 'A1', 'B', 'A2'):
    directory = root/arm
    assert json.loads((directory/'complete.json').read_text())['status'] == 'complete'
    assert not json.loads((directory/f'P8-first-table-{arm}.stop.json').read_text())['remaining']
    plan = json.loads((directory/'plan.json').read_text())
    for name, digest in plan['sources'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest
        sources.add(repo/name)
    files.update(p for p in directory.iterdir() if p.is_file())
files.update(p for p in (root/'B-owner-guard').iterdir() if p.is_file())
files.update(p for p in root.iterdir() if p.is_file() and p.suffix in ('.py', '.json', '.log', '.md'))
files |= sources
files.add(repo/'test/registered/unit/layers/test_dsv4_first_mix_table.py')
manifest = []
with tarfile.open(archive, 'x:gz') as target:
    for path in sorted(files):
        assert path.stat().st_size < 64 * 1024 * 1024, path
        data = path.read_bytes()
        name = str(path.relative_to(repo))
        target.add(path, arcname=name, recursive=False)
        manifest.append(dict(path=name,bytes=len(data),sha256=hashlib.sha256(data).hexdigest()))
record = dict(archive=archive.name,bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=manifest,
              interrupted_B='Stopped before any scored leg: independent pytest PID 3035562 violated GPU exclusivity',
              no_hit_attempt_archive='no-hit-evidence.tar.gz')
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k != 'files'},indent=2))
