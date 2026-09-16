"""Run the integrated first-boundary intervention strictly sequentially."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
proof=json.loads((root/'first-mhc-integrated.json').read_text())
assert proof['status']=='complete' and proof['integrated']
assert len(proof['cases'])==3
for case in proof['cases']:
    assert all(s['bf16_vs_fp32']['byte_exact'] for s in case['shapes'])
script=root/'service.py';digest=hashlib.sha256(script.read_bytes()).hexdigest()
for arm in ('mixed','serial'):
    assert hashlib.sha256(script.read_bytes()).hexdigest()==digest
    with (root/f'trace-v3-{arm}-driver.log').open('x') as log:
        subprocess.run([sys.executable,str(script),'--arm',arm,'--trace','--trace-version','v3'],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    stopped=root/f'trace-v3-{arm}'/f'P32k-tail-policy-trace-v3-{arm}.stop.json'
    assert not json.loads(stopped.read_text())['remaining']
    print('FINISHED',arm,flush=True)
subprocess.run([sys.executable,str(root/'analyze_trace.py'),'--version','v3'],check=True)
