"""Preserve the stopped first32K attempt and its incorrect harness assumption."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
failed=root/'failed-mhc-assumption-A1'
assert not (failed/'complete.json').exists()
assert json.loads((failed/'P16-wide32k-A1.stop.json').read_text())['remaining']==[]
assert [r['leg'] for r in json.loads((failed/'progress.json').read_text())]==['warmup']
files=[(p,str(p.relative_to(root))) for p in sorted(failed.iterdir()) if p.is_file()]
for name in ('failed-run-mhc-assumption.py','failed-mhc-assumption-A1-run.log','failed-mhc-assumption-sweep.log'):
    files.append((root/name,name))
archive=root/'failed-mhc-assumption-evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as f:
    for path,name in files:f.add(path,arcname=name)
manifest=dict(files=len(files),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    scope='Failed after completed warmup due to agent harness requiring mix8 despite legacy batch1 split-K priority. No timed ABBA legs. Preserved by renaming; state absolute paths describe the original pre-rename location.')
(root/'failed-mhc-assumption-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
