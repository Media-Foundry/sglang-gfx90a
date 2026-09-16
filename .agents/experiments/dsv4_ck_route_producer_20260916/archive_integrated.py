"""Archive failed include integration and successful actual-selector validation."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root / 'integrated-evidence.tar.gz'
manifest = root / 'integrated-manifest.json'
assert not target.exists() and not manifest.exists()
files = set()
for version in (1, 2):
    report = root / f'integrated-v{version}.json'
    record = json.loads(report.read_text())
    assert record['status'] == ('failed' if version == 1 else 'complete')
    if version == 1:
        assert "file not found" in record['error']
    else:
        assert len(record['cases']) == 6
        assert all(len(c['checks']) == 4 and all(q['output_byte_exact'] for q in c['checks'])
                   for c in record['cases'])
    for name, digest in record['sources'].items():
        path = Path(name)
        if version == 1 and path.name == 'integrated.py':
            path = root / 'integrated_v1.py'
        if version == 1 and path.name == 'gfx90a_ck_route_producer.cuh':
            path = root / 'integrated_header_v1.cuh'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, path
        files.add(path)
    files.update((report, root / f'integrated-v{version}.log'))
with tarfile.open(target, 'w:gz') as archive:
    for path in sorted(files):
        archive.add(path, arcname=str(path.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(repo)), sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(files)]), indent=2) + '\n')
print('INTEGRATED ARCHIVE', len(files), target.stat().st_size)
