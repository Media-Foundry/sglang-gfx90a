"""Freeze successful confirmation and the preceding fresh-process failure."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
repo=root.parents[2]
r=json.loads((root/'confirm.json').read_text())
assert r['status']=='complete' and r['all_shape_row_exact'] and len(r['numeric'])==270
assert hashlib.sha256((root/'confirm.py').read_bytes()).hexdigest()==r['source_sha256']
files=[root/name for name in ('confirm.py','confirm.json','confirm.log','archive_confirm.py',
    'archive_confirm_failure.py','confirm-no-enumeration-evidence.tar.gz',
    'confirm-no-enumeration.json','confirm-no-enumeration.log','oracle.py','oracle.json')]
files.append(repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py')
archive=root/'confirm-evidence.tar.gz'
entries=[]
with tarfile.open(archive,'x:gz') as out:
    for path in files:
        name=str(path.relative_to(repo))
        out.add(path,arcname=name)
        entries.append(dict(path=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
record=dict(archive=archive.name,bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=entries)
(root/'confirm-archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
