"""Keep all build failures and successful provenance, excluding object/DSO files."""
import hashlib,json,tarfile
from pathlib import Path

root=Path(__file__).resolve().parent;target=root/'build-evidence.tar.gz';assert not target.exists()
paths=[p for d in root.glob('build-*') if d.is_dir() for p in d.rglob('*')
       if p.is_file() and p.suffix in ('.json','.log')]
with tarfile.open(target,'w:gz') as archive:
    for p in sorted(paths):archive.add(p,arcname=str(p.relative_to(root)),recursive=False)
(root/'build-evidence.json').write_text(json.dumps(dict(
    archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),binaries_included=False,
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)}),indent=2)+'\n')
print(target,target.stat().st_size)
