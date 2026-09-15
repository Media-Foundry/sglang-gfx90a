"""Bounded evidence bundle; complete per-row tensors remain local."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
report=json.loads((root/'analysis.json').read_text())
files=set(root.glob('*.py'))|{root/'analysis.json',root/'historical.json'}
for arm in ('A2','B2'):
    base=root/arm
    files.update(base.glob('*.json'));files.update(base.glob('*.sh'))
    files.update(base.glob('*.log'))
    files.update((base/'data').glob('*.json'))
    files.add(base/'data/layer-0-rank-0-metadata.pt')
for item in report['first_observed_differences'][:12]:
    stem=f"layer-{item['layer']}-rank-{item['rank']}-{item['stage']}"
    for arm in ('A2','B2'):
        p=root/arm/'data'/(stem+'.pt')
        if p.exists():files.add(p)
# Retain initial watchdog failure without accepting its request as complete.
files.update((root/'A').glob('*.json'));files.update((root/'A').glob('*.sh'))
files.update((root/'A').glob('*.log'))
manifest={str(p.relative_to(root)):dict(size=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)}
mp=root/'archive-manifest.json';assert not mp.exists()
mp.write_text(json.dumps(dict(files=manifest,full_row_tensors_retained_local=True),indent=2)+'\n')
files.add(mp)
archive=root/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tar:
    for p in sorted(files):tar.add(p,arcname=str(p.relative_to(root)))
assert archive.stat().st_size<90*1024*1024
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'evidence.sha256').write_text(digest+'  evidence.tar.gz\n')
print(archive.stat().st_size,digest)
