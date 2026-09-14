"""Preserve wider-wrapper checks separately from the original direct oracle."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
for name,count in [('supplement.json',4),('integrated.json',3)]:
    d=json.loads((root/name).read_text())
    assert d['status']=='complete' and len(d['cases'])==count
    assert all(c['mutations']==100 and c['fixed_graph_replays']==100 and
               c['score_bits_exact'] and c['logical_physical_exact'] for c in d['cases'])
names=['screen.py','build_inputs.py','validate_inputs.py','metadata.py','test_metadata.py',
       'supplement.json','supplement.log','integrated.json','integrated.log',
       'input-validation.json','input-validation.log','build16k.log','build32k.log',
       'inputs16k/prefill.json','inputs32k/prefill.json']
archive=root/'integrated-evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as f:
    for name in names:f.add(root/name,arcname=name)
manifest=dict(files=len(names),bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              scope='Supplemental direct and opt-in wrapper GPU checks; real input validation; no E2E result yet. Screen source is integrated version; supplemental run records its earlier driver hash.')
(root/'integrated-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
