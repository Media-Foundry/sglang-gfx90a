"""Freeze first fresh-process confirmation failure before adding preflight."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
r=json.loads((root/'confirm.json').read_text())
assert r['status']=='failed' and 'rocBLAS status 6' in r['error']
assert not r['numeric'] and not r['timings']
assert hashlib.sha256((root/'confirm.py').read_bytes()).hexdigest()==r['source_sha256']
with tarfile.open(root/'confirm-no-enumeration-evidence.tar.gz','x:gz') as target:
    for name in ('confirm.py','confirm.json','confirm.log'):
        target.add(root/name,arcname=name)
(root/'confirm.json').rename(root/'confirm-no-enumeration.json')
(root/'confirm.log').rename(root/'confirm-no-enumeration.log')
