"""Resume B/A2 after a pre-measurement GPU ownership abort; preserve evidence."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
repo = root.parents[2]
assert json.loads((root/'A1/complete.json').read_text())['status'] == 'complete'
for arm in ('check', 'A1', 'B'):
    assert not json.loads((root/arm/f'P8-first-table-{arm}.stop.json').read_text())['remaining']
assert not (root/'B/complete.json').exists()
assert not (root/'B/progress.json').exists()
assert 'Unexpected GPU owners: {3035562}' in (root/'B-driver.log').read_text()
sources = json.loads((root/'A1/plan.json').read_text())['sources']
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest() == h for p,h in sources.items())
assert not (root/'B-owner-guard').exists()
assert not (root/'B-owner-guard-driver.log').exists()
(root/'B').rename(root/'B-owner-guard')
(root/'B-driver.log').rename(root/'B-owner-guard-driver.log')
for arm in ('B', 'A2'):
    assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest() == h for p,h in sources.items())
    with (root/(arm+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'service.py'),'--arm',arm],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert not json.loads((root/arm/f'P8-first-table-{arm}.stop.json').read_text())['remaining']
    print('FINISHED', arm, flush=True)
