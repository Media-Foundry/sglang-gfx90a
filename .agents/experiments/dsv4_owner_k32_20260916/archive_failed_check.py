"""Preserve the failed dispatch diagnostic; it did not validate K32 service use."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
folder=root/'check'
assert not (folder/'complete.json').exists()
assert json.loads((folder/'timing-paths.json').read_text())['k32_ranks']==[]
assert not json.loads((folder/'P32-owner-k32-check.stop.json').read_text())['remaining']
driver=(root/'check-driver.log').read_text()
assert 'AssertionError: []' in driver and 'STOPPED P32-owner-k32-check' in driver
target=root/'failed-check-evidence.tar.gz'
manifest=root/'failed-check-manifest.json'
assert not target.exists() and not manifest.exists()
files=[p for p in sorted(folder.rglob('*')) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch','.txt')]
files.append(root/'check-driver.log')
with tarfile.open(target,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
record=dict(status='failed_dispatch_k32_not_selected',tested_source_commit='395b95dd18',
    cause='Initial candidate admitted preshuffle0 only; launcher enables AIter and selects preshuffle16.',
    k32_service_validated=False,throughput_score=False,
    archive=target.name,bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    files=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
manifest.write_text(json.dumps(record,indent=2)+'\n')
print('FAILED CHECK ARCHIVED',len(files),target.stat().st_size)
