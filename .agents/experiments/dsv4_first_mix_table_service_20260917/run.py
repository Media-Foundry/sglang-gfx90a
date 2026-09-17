"""ABBA only after the live eight-rank check, strictly one service at a time."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
assert json.loads((root/'check/complete.json').read_text())['live_exact_checks']>=40
assert not json.loads((root/'check/P8-first-table-check.stop.json').read_text())['remaining']
source=root/'service.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
for arm in ('A1','B','A2'):
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    with (root/(arm+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(source),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert not json.loads((root/arm/f'P8-first-table-{arm}.stop.json').read_text())['remaining']
    print('FINISHED',arm,flush=True)
