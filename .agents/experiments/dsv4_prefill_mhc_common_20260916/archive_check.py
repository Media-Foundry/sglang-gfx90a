"""Preserve live/component checks before the formal service experiment."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
assert json.loads((root/'oracle-v2.json').read_text())['status']=='complete'
assert json.loads((root/'check/complete.json').read_text())['live_comparisons']==10880
target=root/'check-evidence.tar.gz';manifest=root/'check-manifest.json'
assert not target.exists() and not manifest.exists()
files=[p for p in sorted((root/'check').rglob('*')) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files.extend(root/n for n in ('oracle.json','oracle.log','oracle-v2.log','unit.log','check-driver.log'))
with tarfile.open(target,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
manifest.write_text(json.dumps(dict(sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,files=[dict(path=str(p.relative_to(root)),
    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),indent=2)+'\n')
print('Archived checks',len(files),target.stat().st_size)
