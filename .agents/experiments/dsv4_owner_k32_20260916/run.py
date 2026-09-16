"""Serial owned-process ABBA after a separately completed live diagnostic.

Each arm starts fresh except B1/B2, which intentionally share a process.
The service driver owns cleanup in finally. Never launch alongside another run.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
assert json.loads((root/'check/complete.json').read_text())['live_comparisons']==2688
paths=[root/'service.py',root/'analyze.py',Path(__file__)]
sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for arm in ('A1','B','A2'):
    assert sources=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (root/(arm+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'service.py'),'--arm',arm],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('FINISHED',arm,flush=True)
subprocess.run([sys.executable,str(root/'analyze.py')],check=True)
