"""Freeze corrected-model ABBA, quality and marker evidence without binary/tensor caches."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert summary['gain_pct']>3 and all(c['all_twelve_exact'] for c in summary['cross_arm_continuations'])
assert json.loads((root/'profile/analysis.json').read_text())['snapshots']==128
files=set(root.glob('*.py'))|set(root.glob('*.json'))|set(root.glob('*.sh'))
sources=set()
for arm in ('check','A1','B','A2','profile'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    plan=json.loads((directory/'plan.json').read_text())
    for name,digest in plan['sources'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
        sources.add(name)
    for ext in ('*.json','*.log','*.sh'):files.update(directory.glob(ext))
files.update((root/'profile/markers').glob('*.json'))
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name in sorted(sources):archive.add(repo/name,arcname='sources/'+name)
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,evidence_files=len(files),source_files=len(sources),
    full_tensors_or_binaries_included=False)
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
