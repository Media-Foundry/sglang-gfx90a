"""Freeze the bounded component screen; confirmation is separate and pending."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
repo=root.parents[2]
report=json.loads((root/'oracle.json').read_text())
assert report['status']=='complete'
assert hashlib.sha256((root/'oracle.py').read_bytes()).hexdigest()==report['source_sha256']
files=[root/name for name in ('oracle.py','oracle.json','oracle.log','archive_failure.py',
    'index-zero.json','index-zero.log','index-zero-evidence.tar.gz','confirm.py','archive.py')]
files.append(repo/'python/sglang/kernels/ops/attention/dsv4/gfx90a_rocblas_linear.py')
headers=[Path('/opt/rocm/include/rocblas/internal')/name for name in
    ('rocblas-version.h','rocblas-types.h','rocblas-beta.h','rocblas-functions.h')]
entries=[]
archive=root/'evidence.tar.gz'
with tarfile.open(archive,'x:gz') as target:
    for path in [*files,*headers]:
        name=str(path.relative_to(repo)) if path.is_relative_to(repo) else 'installed-headers/'+path.name
        target.add(path,arcname=name)
        entries.append(dict(path=str(path),archive_path=name,bytes=path.stat().st_size,
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
record=dict(archive=archive.name,bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=entries,
    confirmation_status='Script prepared; not executed in this archive',
    failure_note='First run rejected negative IDs using a stale positive-only assumption; no candidate kernels ran')
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({k:v for k,v in record.items() if k!='files'},indent=2))
