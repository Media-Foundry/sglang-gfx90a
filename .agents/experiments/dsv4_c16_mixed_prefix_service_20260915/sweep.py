"""Sequential ABBA; stop on any failed input/cache/ownership gate."""
from pathlib import Path
import subprocess
import sys
root=Path(__file__).resolve().parent
for arm in ('A1','B','A2'):
    assert not (root/arm).exists(), f'Refuse overwrite/restart: {arm}'
    print('ARM START',arm,flush=True)
    with (root/(arm+'-run.log')).open('w') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('ARM COMPLETE',arm,flush=True)
