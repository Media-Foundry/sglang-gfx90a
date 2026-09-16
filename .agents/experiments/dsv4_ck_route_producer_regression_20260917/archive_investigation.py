"""Archive completed ABBA plus failed historical bridge and fresh control.

This deliberately does not create a32K acceptance or erase either failure.
"""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
repo=root.parents[2]
target=root/'investigation-evidence.tar.gz';manifest=root/'investigation-manifest.json'
assert not target.exists() and not manifest.exists()
assert not (root/'acceptance-32k.json').exists()
files=[]
for length in ('16k','32k'):
    for arm in ('A1','B','A2'):
        directory=root/(length+'-'+arm)
        assert (directory/'complete.json').is_file()
        assert not json.loads((directory/f'P{length}-route-producer-regression-{arm}.stop.json').read_text())['remaining']
        plan=json.loads((directory/'plan.json').read_text())
        assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
        files.extend(p for p in directory.iterdir() if p.is_file())
        files.append(root/(length+'-'+arm+'-driver.log'))
for name,label in [('32k-prior-bridge','P32k-route-prior-bridge'),
                   ('32k-prior-control','P32k-route-prior-control')]:
    directory=root/name
    assert not json.loads((directory/(label+'.stop.json')).read_text())['remaining']
    plan=json.loads((directory/'plan.json').read_text())
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in plan['sources'].items())
    files.extend(p for p in directory.iterdir() if p.is_file())
control=json.loads((root/'32k-prior-control/complete.json').read_text())
assert control['teacher_exact'] and control['positions']==1008 and control['route_ranks']==[]
assert not (root/'32k-prior-bridge/complete.json').exists()
assert 'historical teacher logprob mismatch' in (root/'bridge-driver.log').read_text()
files.extend(root/name for name in ('service.py','run.py','run.log','analyze_v1.py',
    'analyze.py','accept.py','bridge_prior_teacher.py','control_teacher.py',
    'bridge-driver.log','control-teacher-driver.log','audit_teacher_lineage.py',
    'teacher-lineage.json','summary-16k.json','acceptance-16k.json'))
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
manifest.write_text(json.dumps(dict(status='32k_acceptance_incomplete',archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest())
           for p in sorted(files)]),indent=2)+'\n')
print('INVESTIGATION ARCHIVE',len(files),target.stat().st_size)
