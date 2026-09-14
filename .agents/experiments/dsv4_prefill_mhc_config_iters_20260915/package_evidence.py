"""Package completed service evidence, retaining raw drift and cold events."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
assert [x['name'] for x in summary['legs']]==['A1','B1','B2','A2']
files=[(root/n,n) for n in ('summary.json','sweep.log','run.py','sweep.py','status.py',
    'analyze.py','test_analysis.py','review_quality.py','quality-review.json',
    'manual-review.md','oracle.py','screen.json','large.json')]
for arm in ('A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert json.loads((directory/f'P16-mhc20-{arm}.stop.json').read_text())['remaining']==[]
    files += [(p,str(p.relative_to(root))) for p in sorted(directory.iterdir())
              if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
    files.append((root/(arm+'-run.log'),arm+'-run.log'))
assert len(files)==len({name for _,name in files})
assert all(p.stat().st_size<16*1024*1024 for p,_ in files)
archive=root/'service-evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for path,name in files:out.add(path,arcname=name)
result=dict(files=len(files),bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    scope='Full C16x32K config20 prefill service ABBA,1M KV; consult summary and manual review, not a global correctness certificate.')
(root/'service-evidence-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
