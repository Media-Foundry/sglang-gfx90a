"""Package bounded compressor evidence, excluding weights and activation tensors."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
run=root/'stable-layer2-compressor'
assert (run/'complete.json').exists()
assert json.loads((run/'INPUT-IDENTITY.stop.json').read_text())['remaining']==[]
summary=json.loads((run/'compressor-summary.json').read_text())['comparisons']
assert all(v['changed_elements']==0 for v in summary['A1-A2'].values())
assert summary['A1-B1']['core_input']['changed_elements']==0
assert summary['A1-B1']['core_result']['changed_elements']==1
replay=json.loads((run/'service-projection-oracle.json').read_text())
assert len([r for r in replay['results'] if r.get('full_service_reproduced')])==6
files=[p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files += [root/n for n in ('stable-layer2-compressor-run.log','compressor-service-oracle.log',
                         'compressor-projection-oracle.json')]
out=root/'compressor-evidence.tar.gz'
assert not out.exists()
with tarfile.open(out,'w:gz') as archive:
    for p in files: archive.add(p,arcname=str(p.relative_to(root)))
manifest=dict(files=len(files),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(root/'compressor-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
