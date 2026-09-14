"""Archive completed 43-layer sampled diagnostics, never full tensor dumps."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
run=root/'stable-all-layers'
target=root/'stable-all-layer-evidence.tar.gz'
assert (run/'complete.json').exists() and not target.exists()
data=json.loads((run/'all-layer-summary.json').read_text())
assert len(data['layers'])==43 and data['input_identity_exact'] and data['sampled_only']
assert data['first_boundary_difference']['layer']==2
assert data['first_control_boundary_difference']['layer']==15
assert all(len(l['metadata'])==8 for l in data['layers'])
outputs={n:json.loads((run/(n+'-by-case.json')).read_text()) for n in ('A1','B1','A2')}
assert all(len(r)==16 for r in outputs.values())
for responses in outputs.values():
    assert all(len(r['output_ids'])==128 and r['meta_info']['completion_tokens']==128 for r in responses)
quality={a+'-'+b:sum(x['output_ids']==y['output_ids'] for x,y in zip(outputs[a],outputs[b],strict=True))
         for a,b in (('A1','A2'),('A1','B1'))}
(run/'output-repeat-summary.json').write_text(json.dumps(quality,indent=2)+'\n')
files=[p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files += [root/n for n in ('stable-all-layers-run.log','stable-all-layers-analysis.log','router-membership-audit.json')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
manifest=dict(files=len(files),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              output_matches=quality)
(root/'stable-all-layer-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
