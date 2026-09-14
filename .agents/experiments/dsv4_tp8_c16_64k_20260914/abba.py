"""Run fresh-process A1, B1/B2, A2; each arm owns and stops its service."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
assert (root/'pilot/complete.json').exists()
assert all(not (root/arm).exists() for arm in ('A1','B','A2'))
for arm in ('A1','B','A2'):
    print('START ARM',arm,flush=True)
    with (root/(arm+'-run.log')).open('w') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    print('COMPLETE ARM',arm,flush=True)
