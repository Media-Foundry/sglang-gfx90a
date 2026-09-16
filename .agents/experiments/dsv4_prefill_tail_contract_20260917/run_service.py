"""Sequential teacher diagnostics; no performance scoring or overlapping runs."""
import hashlib
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
source=root/'service.py';digest=hashlib.sha256(source.read_bytes()).hexdigest()
for arm in ('baseline','mixed','serial'):
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    with (root/(arm+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(source),'--arm',arm],stdout=log,
                       stderr=subprocess.STDOUT,check=True)
    print('FINISHED',arm,flush=True)
