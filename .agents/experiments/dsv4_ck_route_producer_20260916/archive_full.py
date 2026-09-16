"""Preserve full-stage failures and successful revisions with exact source snapshots."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
target=root/'full-stage-evidence.tar.gz';manifest=root/'full-stage-manifest.json'
assert not target.exists() and not manifest.exists()
reports={}
snapshots={1:'full_stage_v1.py',2:'full_stage_v2.py',3:'full_stage.py'}
for version,snapshot in snapshots.items():
    result=json.loads((root/f'full-stage-v{version}.json').read_text())
    assert result['sources'][str(root/'full_stage.py')]==hashlib.sha256((root/snapshot).read_bytes()).hexdigest()
    assert result['sources'][str(root/'runner.py')]==hashlib.sha256((root/'runner.py').read_bytes()).hexdigest()
    assert result['status']==('failed' if version==1 else 'complete')
    if version==1:
        assert 'tensor model parallel group is not initialized' in result['error']
        assert result['cases']==[]
    else:
        assert len(result['cases'])==(3 if version==2 else 4)
        assert all(c['expanded_weights_exact'] and len(c['checks'])==4
                   and all(q['output_byte_exact'] for q in c['checks']) for c in result['cases'])
    reports[str(version)]=dict(status=result['status'],source_snapshot=snapshot)
files=[root/f'full-stage-v{version}.{ext}' for version in snapshots for ext in ('json','log')]
files += [root/name for name in (*snapshots.values(),'runner.py')]
with tarfile.open(target,'w:gz') as archive:
    for path in files:archive.add(path,arcname=str(path.relative_to(root)))
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
    revisions=reports,files=[dict(path=str(p.relative_to(root)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('FULL STAGE ARCHIVE',len(files),target.stat().st_size)
