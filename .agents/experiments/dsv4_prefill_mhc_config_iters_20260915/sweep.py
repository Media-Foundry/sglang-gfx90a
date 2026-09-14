"""No implicit overwrite/restart of an owned fresh-process MHC-policy ABBA."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for arm in ('A1','B','A2'):
    assert not (root/arm).exists(),f'Refuse to overwrite {arm}'
    print('ARM START',arm,flush=True)
    with (root/(arm+'-run.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    print('ARM COMPLETE',arm,flush=True)
