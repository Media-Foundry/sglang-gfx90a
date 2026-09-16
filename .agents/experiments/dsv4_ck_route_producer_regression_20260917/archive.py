"""Preserve completed long-input ABBA, regardless of performance acceptance."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
repo=root.parents[2]
target=root/'regression-evidence.tar.gz'
manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
files=[]
for length in ('16k','32k'):
    summary=root/('summary-'+length+'.json')
    assert json.loads(summary.read_text())['status']=='complete'
    files.append(summary)
    for arm in ('A1','B','A2'):
        directory=root/(length+'-'+arm)
        assert (directory/'complete.json').is_file()
        assert not json.loads((directory/f'P{length}-route-producer-regression-{arm}.stop.json').read_text())['remaining']
        plan=json.loads((directory/'plan.json').read_text())
        assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
        files.extend(p for p in directory.iterdir() if p.is_file())
        files.append(root/(length+'-'+arm+'-driver.log'))
    for name in ('acceptance-'+length+'.json','acceptance-'+length+'.log'):
        if (root/name).is_file():files.append(root/name)
files.extend(root/name for name in ('service.py','analyze.py','accept.py','run.py','run.log'))
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
manifest.write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(files)]),indent=2)+'\n')
print('REGRESSION ARCHIVE',len(files),target.stat().st_size)
