"""Keep partial screen and LDS-resource failure before completing remaining cases."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
record=json.loads((root/'qkv-fixed-screen.json').read_text())
assert record['status']=='failed' and len(record['cases'])==7
assert 'OutOfResources' in record['error'] and '81920' in record['error']
assert hashlib.sha256((root/'qkv_fixed_screen.py').read_bytes()).hexdigest()==record['source_sha256']
target=root/'qkv-screen-partial-evidence.tar.gz';assert not target.exists()
names=('qkv_fixed_screen.py','qkv-fixed-screen.json','qkv-fixed-screen.log','archive_qkv_screen_failure.py')
with tarfile.open(target,'w:gz') as archive:
    for name in names:archive.add(root/name,arcname=name)
(root/'qkv-screen-partial-manifest.json').write_text(json.dumps(dict(archive=target.name,
    sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names}),indent=2)+'\n')
print('ARCHIVED PARTIAL SCREEN',target.stat().st_size)
