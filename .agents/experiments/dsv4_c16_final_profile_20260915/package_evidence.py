"""Archive completed v2 markers and separately labeled failed first attempt."""
import hashlib
import json
from pathlib import Path
import tarfile
import argparse
import re

root = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--label',help='Package a new standalone completed capture, without historical failed runs.')
parser.add_argument('--driver',default='capture.py')
parser.add_argument('--report',default='current-mix8.md')
args=parser.parse_args()
assert args.label is None or re.fullmatch(r'[A-Za-z0-9-]+',args.label)
assert re.fullmatch(r'[A-Za-z0-9_]+\.py',args.driver)
assert re.fullmatch(r'[A-Za-z0-9_-]+\.md',args.report)
accepted = root/(args.label or 'capture-v2')
assert json.loads((accepted/'complete.json').read_text())['frames'] == 128
assert json.loads((accepted/'analysis.json').read_text())['snapshots'] == 128
assert (accepted/'details-analysis.json').exists()
assert json.loads((accepted/'P16-markers-B.stop.json').read_text())['remaining'] == []
directories=[(accepted,'accepted')]
if args.label is None:
    failed = root/'capture'
    assert not (failed/'complete.json').exists()
    assert json.loads((failed/'P16-markers-B.stop.json').read_text())['remaining'] == []
    directories.append((failed,'failed-harness'))
files = []
for directory, label in directories:
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.log', '.sh', '.patch'):
            assert p.stat().st_size < 16*1024*1024
            files.append((p, label+'/'+str(p.relative_to(directory))))
names=[args.driver,'analyze.py','path_checks.py','test_path_checks.py','package_evidence.py']
if args.label is None:names+=['capture.log','capture-v2.log','analysis-v2.log']
else:names+=[args.report]
for name in names:
    files.append((root/name, name))
archive = root/(f'{args.label}-evidence.tar.gz' if args.label else 'evidence.tar.gz')
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as out:
    for p, name in files:
        out.add(p, arcname=name)
manifest = dict(files=len(files), bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(path=name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in files])
(root/(f'{args.label}-evidence-manifest.json' if args.label else 'evidence-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k != 'members'})
