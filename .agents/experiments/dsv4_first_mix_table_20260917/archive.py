"""Archive the first-mix table oracle source and measured proof, not model weights."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';manifest=root/'archive-manifest.json'
assert not target.exists() and not manifest.exists()
proof=json.loads((root/'oracle.json').read_text())
assert proof['status']=='complete' and proof['whole_vocab_native_variants_exact'] and proof['service_mixes_exact']
assert len(proof['checks'])==7 and all(c['graph_mutations_exact']==100 for c in proof['checks'])
files={root/'oracle.json',root/'oracle.log',root/'archive.py'}
for name,digest in proof['sources'].items():
    p=repo/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==digest
    files.add(p)
with tarfile.open(target,'w:gz') as archive:
    for p in sorted(files):archive.add(p,arcname=str(p.relative_to(repo)))
manifest.write_text(json.dumps(dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(repo)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]),indent=2)+'\n')
print('ARCHIVED',len(files),target.stat().st_size)
