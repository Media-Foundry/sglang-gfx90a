"""Archive MHC post arithmetic diagnosis, exact HIP checks and service ABBA."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent;repo=root.parents[2]
target=root/'evidence.tar.gz';assert not target.exists()
summary=json.loads((root/'summary.json').read_text())
assert summary['post_exact_comparisons']==2720
assert all(c['all_twelve_exact'] for c in summary['cross_arm_continuations'])
assert json.loads((root/'validation.json').read_text())['status']=='complete'
files=set(root.glob('*.py'))|set(root.glob('*.json'))|set(root.glob('*.cuh'))|set(root.glob('*.sh'))
sources=set()
for arm in ('check','A1','B','A2'):
    directory=root/arm
    assert (directory/'complete.json').exists()
    assert not json.loads((directory/f'P16-post-wave-{arm}.stop.json').read_text())['remaining']
    for name,digest in json.loads((directory/'plan.json').read_text())['sources'].items():
        assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
        sources.add(name)
    for pattern in ('*.json','*.log','*.sh'):files.update(directory.glob(pattern))
for directory in ('compiler','hip-compiler'):
    files.update(p for p in (root/directory).iterdir() if p.suffix in ('.json','.txt','.amdgcn','.llir','.ttir','.ttgir'))
with tarfile.open(target,'w:gz') as archive:
    for path in sorted(files):archive.add(path,arcname=str(path.relative_to(root)))
    for name in sorted(sources):archive.add(repo/name,arcname='sources/'+name)
record=dict(archive=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),bytes=target.stat().st_size,
    evidence_files=len(files),source_files=len(sources),full_tensors_or_binaries_included=False)
(root/'archive-manifest.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
