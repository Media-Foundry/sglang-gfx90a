"""Sequential 8K and 16K regression ABBA with owned cleanup, no overlapping services."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
accepted=json.loads((root.parent/'dsv4_owner_k32_20260916/acceptance.json').read_text())
assert accepted['status']=='accepted_explicit_32k_profile' and accepted['numerical_exact_on_tested_inputs']
paths=[root/name for name in ('service.py','analyze.py','run.py')]
frozen={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for length in ('8k','16k'):
    for arm in ('A1','B','A2'):
        assert frozen=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        with (root/(length+'-'+arm+'-driver.log')).open('x') as log:
            subprocess.run([sys.executable,str(root/'service.py'),'--length',length,'--arm',arm],
                           stdout=log,stderr=subprocess.STDOUT,check=True)
        print('FINISHED',length,arm,flush=True)
    subprocess.run([sys.executable,str(root/'analyze.py'),'--length',length],check=True)
