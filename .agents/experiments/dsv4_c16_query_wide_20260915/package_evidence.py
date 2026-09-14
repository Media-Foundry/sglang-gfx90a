"""Archive the first validated wider-C4 component suite, including failed oracle."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
full=json.loads((root/'full.json').read_text())
assert full['status']=='complete' and len(full['cases'])==4
assert all(c['mutations']==100 and c['fixed_graph_replays']==100 and
           c['score_bits_exact'] and c['logical_physical_exact'] for c in full['cases'])
names=('screen.py','failed_screen_source.py','metadata.py','test_metadata.py','build_inputs.py',
       'screen.json','screen.log','screen-v2.json','screen-v2.log','full.json','full.log')
archive=root/'component-evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for name in names:out.add(root/name,arcname=name)
manifest=dict(files=len(names),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    scope='Standalone wider logits/Top-K only; production guard unchanged; first screen failed due to ascending tie oracle')
(root/'component-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
