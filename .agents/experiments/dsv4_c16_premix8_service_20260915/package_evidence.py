"""Package completed query-reuse ABBA; never archive a partial run as accepted."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
assert [x['name'] for x in summary['legs']]==['A1','B1','B2','A2']
files=[(root/'summary.json','summary.json'),(root/'sweep.log','sweep.log')]
files.append((root/'case8-review.json','case8-review.json'))
for arm in ('A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert json.loads((directory/f'P16-mix8-{arm}.stop.json').read_text())['remaining']==[]
    files += [(p,str(p.relative_to(root))) for p in sorted(directory.iterdir())
              if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
    files.append((root/(arm+'-run.log'),arm+'-run.log'))
for name in ('screen.json','screen.log','full.json','full.log','integrated.json','integrated.log'):
    files.append((root.parent/'dsv4_c16_premix8_20260915'/name,'component/'+name))
assert len({name for _,name in files})==len(files)
assert all(p.stat().st_size<16*1024*1024 for p,_ in files)
archive=root/'evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p,name in files:out.add(p,arcname=name)
manifest=dict(files=len(files),bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
