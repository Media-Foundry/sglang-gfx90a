"""Freeze completed tail-policy diagnostics before adding new probes."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root / 'service-evidence.tar.gz'
manifest = root / 'service-manifest.json'
assert not target.exists() and not manifest.exists()
analysis = json.loads((root / 'service-analysis.json').read_text())
assert analysis['status'] == 'diagnostic_complete'
assert not analysis['scored_performance']
files = {root / name for name in (
    'service.py', 'run_service.py', 'analyze_service.py', 'service-analysis.json',
    'analysis-service.log', 'run-service.log', 'archive_service.py')}
for arm in ('baseline', 'mixed', 'serial'):
    directory = root / arm
    assert not json.loads((directory / f'P32k-tail-policy-{arm}.stop.json').read_text())['remaining']
    sources = json.loads((directory / 'plan.json').read_text())['sources']
    for name, digest in sources.items():
        path = repo / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        files.add(path)
    files.update(p for p in directory.rglob('*') if p.is_file())
    files.add(root / f'{arm}-driver.log')
files = sorted(files)
records = [dict(path=str(p.relative_to(repo)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(target, 'w:gz') as archive:
    for path in files:
        archive.add(path, arcname=str(path.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(), files=records,
    status='diagnostic_complete_residual_drift_remaining', scored_performance=False), indent=2)+'\n')
print('ARCHIVED', len(files), 'files', target.stat().st_size, 'bytes')
