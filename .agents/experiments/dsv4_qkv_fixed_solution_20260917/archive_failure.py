"""Preserve the first run: legacy positive-only guard rejected negative IDs.

The historical filename/error says index-zero, but returned IDs were negative,
not zero. Subsequent pinned-source audit establishes their explicit Tensile meaning.
"""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
report=json.loads((root/'oracle.json').read_text())
assert report['status']=='failed' and 'Index zero' in report['error']
assert not report['screen']
files=[root/name for name in ('oracle.py','oracle.json','oracle.log')]
assert hashlib.sha256(files[0].read_bytes()).hexdigest()==report['source_sha256']
with tarfile.open(root/'index-zero-evidence.tar.gz','x:gz') as archive:
    for path in files:archive.add(path,arcname=path.name)
(root/'oracle.json').rename(root/'index-zero.json')
(root/'oracle.log').rename(root/'index-zero.log')
