"""Archive updated post-wave profile and standalone pre-mix experiments."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
parent=root.parent/'dsv4_mhc_post_tiles_20260916';profile=parent/'profile'
assert json.loads((profile/'complete.json').read_text())['frames']==128
assert not json.loads((profile/'P16-markers-postwave.stop.json').read_text())['remaining']
assert json.loads((profile/'analysis.json').read_text())['snapshots']==128
assert json.loads((root/'screen.json').read_text())['status']=='complete'
assert json.loads((root/'boundary.json').read_text())['status']=='complete'
sources=set()
for name,digest in json.loads((profile/'plan.json').read_text())['sources'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
    sources.add(name)
files=set(root.glob('*.py'))|set(root.glob('*.json'))
files.update(p for directory in ('memory','issue') for p in (root/directory).rglob('*.csv'))
profile_files=[p for p in profile.rglob('*') if p.is_file() and p.suffix in ('.json','.log','.sh')]
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for path in sorted(profile_files):archive.add(path,arcname='postwave-profile/'+str(path.relative_to(profile)))
    for name in ('profile.py','analyze_profile.py'):archive.add(parent/name,arcname='profile-scripts/'+name)
    for name in sorted(sources):archive.add(repo/name,arcname='sources/'+name)
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
    experiment_files=len(files),profile_files=len(profile_files),source_files=len(sources),full_tensors_or_binaries_included=False)
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
