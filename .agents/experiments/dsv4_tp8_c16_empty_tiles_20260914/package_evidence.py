"""Bundle completed ABBA evidence, preserving the excluded tooling-failure arm."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
assert [x['name'] for x in summary['legs']]==['A1','B1','B2','A2']
files=[root/'summary.json']
for arm in ('A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert json.loads((directory/f'P16-empty-{arm}.stop.json').read_text())['remaining']==[]
    files += [p for p in sorted(directory.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
    files.append(root/(arm+'-run.log'))
failed=root/'B-import-shadow-failed'
assert failed.exists() and not (failed/'complete.json').exists()
files += [p for p in sorted(failed.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files += [root/n for n in ('B-import-shadow-failed-run.log','sweep.log','sweep-resume-B.log')]
out=root/'evidence.tar.gz';assert not out.exists()
with tarfile.open(out,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
manifest=dict(files=len(files),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
