"""Freeze scalar/vec4 ABBA, recovered post-test harness failure and exact oracles."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert all(c['all_twelve_exact'] for c in summary['cross_arm_continuations'])
files=set(root.glob('*.py'))|set(root.glob('*.json'))
sources={}
for arm in ('A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert not json.loads((directory/f'P16-vec4-reduce-{arm}.stop.json').read_text())['remaining']
    plan=json.loads((directory/'plan.json').read_text())
    for name,digest in plan['sources'].items():
        path=repo/name
        if name==str((root/'service.py').relative_to(repo)) and hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            path=root/'service_measured.py'
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
        sources[name]=path
    for pattern in ('*.json','*.log','*.sh'):files.update(directory.glob(pattern))
files.add(root/'vector.cuh')
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name,path in sorted(sources.items()):archive.add(path,arcname='sources/'+name)
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
    bytes=target.stat().st_size,evidence_files=len(files),source_files=len(sources),
    recovered_B_post_test_log_assertion=True,full_tensors_or_binaries_included=False)
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
