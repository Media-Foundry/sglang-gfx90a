"""Bounded evidence, including selected check payloads; no model weights/binaries."""
import hashlib,json,tarfile
from pathlib import Path

root=Path(__file__).resolve().parent;target=root/'evidence.tar.gz';assert not target.exists()
paths=[]
for name in ('check','A1','B','A2'):
    for p in (root/name).rglob('*'):
        if not p.is_file():continue
        if p.suffix in ('.json','.sh','.log','.patch') or (p.suffix=='.pt' and
            (p.name=='layer-0-rank-0-metadata.pt' or p.name.startswith('layer-1-rank-6-'))):
            paths.append(p)
with tarfile.open(target,'w:gz') as out:
    for p in sorted(paths):out.add(p,arcname=str(p.relative_to(root)),recursive=False)
(root/'evidence.json').write_text(json.dumps(dict(
    archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    all_tensors_included=False,
    files={str(p.relative_to(root)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(paths)}),indent=2)+'\n')
print(target,target.stat().st_size)
