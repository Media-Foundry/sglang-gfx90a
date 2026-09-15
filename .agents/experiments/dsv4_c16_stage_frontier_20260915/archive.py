"""Archive bounded evidence; retain large row-hash/sample tensors locally by SHA."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
assert (root/'analysis.json').exists()
manifest = []
included = []
for arm in ('A', 'B'):
    assert (root/arm/'complete.json').exists()
    for path in sorted((root/arm/'data').glob('*.pt')):
        manifest.append(dict(path=str(path.relative_to(root)), size=path.stat().st_size,
                             sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    included.extend((root/arm).glob('*.json'))
    included.extend((root/arm).glob('*.sh'))
    included.extend((root/arm).glob('*.log'))
    included.extend((root/arm/'data').glob('*.json'))
    included.extend((root/arm/'data').glob('layer-*-rank-0-metadata.pt'))
    included.append(root/f'{arm}-run.log')
(root/'tensor-manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
included.extend([root/'analysis.json', root/'tensor-manifest.json'])
target = root/'evidence.tar.gz'
assert not target.exists()
with tarfile.open(target, 'w:gz') as archive:
    for path in sorted(set(included)):
        archive.add(path, arcname=str(path.relative_to(root)))
assert target.stat().st_size < 90_000_000
result = dict(files=len(set(included)), bytes=target.stat().st_size,
              sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
              tensor_files_local_only=len(manifest),
              note='Complete stage hashes and analysis included; numeric samples and row hashes '
                   'retained locally with tensor-manifest SHA256. No performance measurement.')
(root/'archive.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
