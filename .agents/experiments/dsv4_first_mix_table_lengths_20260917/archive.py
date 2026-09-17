"""Freeze one finished length, its raw responses and exact runtime sources."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
parser = argparse.ArgumentParser()
parser.add_argument('--length', choices=('16k', '32k'), required=True)
args = parser.parse_args()
directory = root/args.length
archive = directory/'evidence.tar.gz'
assert not archive.exists()
assert json.loads((directory/'summary.json').read_text())['status'] == 'complete'
files = {directory/'summary.json'}
files.update(root.glob('*.py'))
files.update(root.glob(f'{args.length}-*.log'))
for arm in ('A1', 'B', 'A2'):
    d = directory/arm
    assert json.loads((d/'complete.json').read_text())['status'] == 'complete'
    assert not json.loads((d/f'P{args.length}-first-table-{arm}.stop.json').read_text())['remaining']
    files.update(p for p in d.iterdir() if p.is_file())
    for name, digest in json.loads((d/'plan.json').read_text())['sources'].items():
        path = repo/name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
        files.add(path)
entries = []
with tarfile.open(archive, 'x:gz') as target:
    for path in sorted(files):
        assert path.stat().st_size < 64*1024*1024
        name = str(path.relative_to(repo))
        target.add(path, arcname=name, recursive=False)
        entries.append(dict(path=name, bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
record = dict(archive=archive.name, bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=entries)
(directory/'archive-manifest.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k != 'files'}, indent=2))
