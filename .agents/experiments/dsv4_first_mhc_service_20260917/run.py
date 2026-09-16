"""Strictly serial A1/B1/B2/A2 service trial; no restart on observation timeout."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent;source=root/'service.py'
digest=hashlib.sha256(source.read_bytes()).hexdigest()
for arm in ('A1','B','A2'):
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    with (root/(arm+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(source),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert not json.loads((root/arm/f'P8-first-mhc-{arm}.stop.json').read_text())['remaining']
    print('FINISHED',arm,flush=True)
