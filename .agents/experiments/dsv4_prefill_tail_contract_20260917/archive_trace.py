"""Archive completed row-aligned probes and their exact source versions."""
import hashlib
import argparse
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--version',choices=('v1','v2','v3'),default='v1')
args=parser.parse_args();prefix='trace-' if args.version=='v1' else f'trace-{args.version}-'
target=root/(prefix+'evidence.tar.gz');manifest=root/(prefix+'manifest.json')
assert not target.exists() and not manifest.exists()
assert json.loads((root/(prefix+'analysis.json')).read_text())['status']=='complete'
files={root/'analyze_trace.py',root/(prefix+'analysis.json'),root/'archive_trace.py'}
for arm in ('mixed','serial'):
    d=root/f'{prefix}{arm}'
    assert not json.loads((d/f'P32k-tail-policy-{prefix}{arm}.stop.json').read_text())['remaining']
    for name,digest in json.loads((d/'plan.json').read_text())['sources'].items():
        p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest,name
        files.add(p)
    files.update(p for p in d.rglob('*') if p.is_file())
    files.add(root/f'{prefix}{arm}-driver.log')
if args.version=='v2':
    files.update(root/name for name in ('first_mhc_oracle.py','first-mhc-oracle.json','first-mhc-oracle.log'))
    for name,digest in json.loads((root/'first-mhc-oracle.json').read_text())['sources'].items():
        p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
        files.add(p)
if args.version=='v3':
    files.update(root/name for name in ('first_mhc_oracle.py','first-mhc-integrated.json',
        'first-mhc-integrated.log','run_trace_pair.py','run-trace-v3.log',
        'compare_trace_teachers.py','trace-teacher-comparison.json',
        'qkv_shape_oracle.py','qkv-shape-oracle.json','qkv-shape-oracle.log'))
    for result in ('first-mhc-integrated.json','qkv-shape-oracle.json'):
        for name,digest in json.loads((root/result).read_text())['sources'].items():
            p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
            files.add(p)
files=sorted(files)
records=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
with tarfile.open(target,'w:gz',compresslevel=1) as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,files=records,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),scored_performance=False),indent=2)+'\n')
print('TRACE ARCHIVED',len(files),target.stat().st_size)
