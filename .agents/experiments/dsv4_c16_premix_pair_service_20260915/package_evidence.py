"""Archive only a complete, stopped trial with explicitly bounded manual review."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
repo = root.parents[2]
archive = root/'service-evidence.tar.gz'
assert not archive.exists()
summary = json.loads((root/'summary.json').read_text())
review = json.loads((root/'quality-review.json').read_text())
assert summary['formal_input_echoes'] == 192
assert review['manual_review_completed'] and review['exact_echoes'] == 96
assert (root/'manual-review.md').is_file()
for arm in ('A1', 'B', 'A2'):
    assert (root/arm/'complete.json').is_file()
    assert json.loads((root/arm/f'P16-mix-pair-{arm}.stop.json').read_text())['remaining'] == []
for name,digest in summary['sources'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest() == digest, name
for name,digest in review['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
files = {p for p in root.iterdir() if p.is_file() and p.suffix in ('.py','.md','.json','.log')}
for arm in ('A1', 'B', 'A2'):
    files.update(p for p in (root/arm).iterdir() if p.is_file() and p.suffix in ('.json','.log','.sh','.patch'))
files.update(repo/name for name in summary['sources'])
for name in ('integrated.py', 'integrated.json'):
    files.add(root.parent/'dsv4_c16_premix_pair_20260915'/name)
assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(repo) for p in files)
with tarfile.open(archive, 'w:gz') as tar:
    for p in sorted(files): tar.add(p, arcname=str(p.relative_to(repo)), recursive=False)
result = dict(files=len(files), bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              scope=summary['metric'], bounded_review_only=True)
(root/'service-evidence-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
