"""Validate owned cleanup/source identity and archive all three diagnostic profiles."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
repo=root.parents[2]
assert json.loads((root/'length-comparison.json').read_text())['diagnostic_only']
target=root/'profile-evidence.tar.gz'; manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[]
for length,n in (('8k',4),('16k',8),('32k',16)):
    folder=root/length
    assert not json.loads((folder/f'P{length}-markers-common.stop.json').read_text())['remaining']
    plan=json.loads((folder/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
    done=json.loads((folder/'complete.json').read_text())
    assert done['diagnostic_only'] and done['input_echo_exact']==64 and done['frames']==32*n
    analysis=json.loads((folder/'analysis.json').read_text())
    assert analysis['snapshots']==32*n and analysis['warmup_frames']==8*n
    files.extend(p for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch','.txt'))
with tarfile.open(target,'w:gz') as archive:
    for p in files: archive.add(p,arcname=str(p.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('PROFILE ARCHIVE',len(files),target.stat().st_size)
