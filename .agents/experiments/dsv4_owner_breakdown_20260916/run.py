"""Serial real8K/32K diagnostic services, no scoring or concurrent GPU jobs."""
import hashlib
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
paths=[root/name for name in ('profile.py','analyze.py','run.py')]
hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
for length,n in (('8k',4),('32k',16)):
    assert hashes=={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with (root/(length+'-driver.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'profile.py'),'--length',length],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(root.parent/'dsv4_prefill_length_profile_20260916/analyze.py'),
                        '--root',str(root/length),'--stop-label','P'+length+'-markers-common',
                        '--forwards-per-wave',str(n)],stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,str(root/'analyze.py'),'--root',str(root/length)],stdout=log,stderr=subprocess.STDOUT,check=True)
    print('OWNER PROFILE COMPLETE',length,flush=True)
