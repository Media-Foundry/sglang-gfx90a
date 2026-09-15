import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for arm in ('A','B'):
    assert not (root/arm).exists()
    print('ARM START',arm,flush=True)
    with (root/(arm+'-run.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-late-drift-{arm}.stop.json').read_text())['remaining']==[]
    print('ARM COMPLETE',arm,flush=True)
