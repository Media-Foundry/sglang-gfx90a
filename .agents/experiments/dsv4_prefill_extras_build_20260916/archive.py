"""Archive rebuild evidence, sources and counters without DSOs, objects or tensors."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert abs(summary['gain_pct'])<1
assert all(c['all_twelve_exact'] for c in summary['cross_arm_continuations'])
files=set(root.glob('*.py'))|set(root.glob('*.json'))
sources=set()
for arm in ('check','A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    plan=json.loads((directory/'plan.json').read_text())
    for name,digest in plan['sources'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
        sources.add(name)
    for pattern in ('*.json','*.log','*.sh'):files.update(directory.glob(pattern))
for path in (root/'build-v1').rglob('*'):
    if path.is_file() and path.suffix in ('.json','.log','.d','.hpp','.env'):files.add(path)
counters=root.parent/'dsv4_ck_counters_20260916'
counter_files=[p for p in counters.rglob('*') if p.is_file() and p.suffix in ('.json','.py','.csv')]
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name in sorted(sources):archive.add(repo/name,arcname='sources/'+name)
    for path in sorted(counter_files):archive.add(path,arcname='counters/'+str(path.relative_to(counters)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,evidence_files=len(files),source_files=len(sources),
    counter_files=len(counter_files),full_tensors_or_binaries_included=False)
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
