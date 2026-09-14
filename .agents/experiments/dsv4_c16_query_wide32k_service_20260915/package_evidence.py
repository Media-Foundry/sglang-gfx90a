"""Archive only the completed wide-C4 service ABBA, with measured drift intact."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
summary=json.loads((root/'summary.json').read_text())
assert [x['name'] for x in summary['legs']]==['A1','B1','B2','A2']
files=[(root/name,name) for name in ('summary.json','sweep.log','run.py','sweep.py',
                                    'analyze.py','status.py','start-base.sh',
                                    'review_quality.py','quality-review.json','manual-review.md')]
for arm in ('A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert json.loads((directory/f'P16-wide32k-{arm}.stop.json').read_text())['remaining']==[]
    files += [(p,str(p.relative_to(root))) for p in sorted(directory.iterdir())
              if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
    files.append((root/(arm+'-run.log'),arm+'-run.log'))
component=root.parent/'dsv4_c16_query_wide_20260915'
for name in ('component-evidence-manifest.json','integrated-evidence-manifest.json'):
    files.append((component/name,'component/'+name))
assert len({name for _,name in files})==len(files)
assert all(p.stat().st_size<16*1024*1024 for p,_ in files)
archive=root/'evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive,'w:gz') as out:
    for p,name in files:out.add(p,arcname=name)
manifest=dict(files=len(files),bytes=archive.stat().st_size,
              sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
              scope='Completed C16x32K zero-prefix service ABBA, 1M KV. Inspect summary for drift and performance; not a bitwise-determinism claim.')
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
