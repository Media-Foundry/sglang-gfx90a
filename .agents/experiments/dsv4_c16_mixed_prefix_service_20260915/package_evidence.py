"""Package only a complete, stopped and explicitly reviewed mixed-prefix trial."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
archive=ROOT/'service-evidence.tar.gz'
assert not archive.exists()
summary=json.loads((ROOT/'summary.json').read_text())
review=json.loads((ROOT/'quality-review.json').read_text())
assert review['bounded_review_completed'] and review['input_echo_exact']==96
assert (ROOT/'manual-review.md').is_file()
for arm in ('A1','B','A2'):
    assert (ROOT/arm/'complete.json').is_file()
    stop=json.loads((ROOT/arm/f'P16-prefix-{arm}.stop.json').read_text())
    assert stop['remaining']==[]
for name,digest in summary['sources'].items():
    assert hashlib.sha256((REPO/name).read_bytes()).hexdigest()==digest, name
for name,digest in review['source_sha256'].items():
    assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
files=set()
for path in ROOT.iterdir():
    if path.is_file() and path.suffix in ('.py','.md','.json','.log'):
        files.add(path)
for arm in ('A1','B','A2'):
    for path in (ROOT/arm).iterdir():
        if path.is_file() and path.suffix in ('.json','.log','.sh','.patch'):
            files.add(path)
files.update(REPO/name for name in summary['sources'])
files.update([ROOT.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py',
              ROOT.parent/'dsv4_c16_query_wide_20260915/test_prefix.py'])
assert all(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(REPO) for p in files)
with tarfile.open(archive,'w:gz') as tar:
    for path in sorted(files):tar.add(path,arcname=str(path.relative_to(REPO)),recursive=False)
result=dict(files=len(files),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    scope=summary['scope'],bounded_review_only=True)
(ROOT/'service-evidence-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
