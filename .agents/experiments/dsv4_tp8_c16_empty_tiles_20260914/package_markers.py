"""Package bounded diagnostic evidence, never model weights or large workspaces."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
files = [root/'markers-B-run.log']
files += sorted((root/'markers-B').glob('*.json'))
files += sorted((root/'markers-B').glob('*.log'))
files += sorted((root/'markers-B').glob('*.sh'))
files += sorted((root/'markers-B').glob('*.patch'))
files += sorted((root/'markers-B'/'markers').glob('*.json'))
for name in ('marker-smoke-v3',):
    files += sorted((root/name).glob('*.json'))
files += [root/'marker-smoke-v3.log']
# Failed profiler attempt stays separately identifiable and is not a timing arm.
for name in ('abort-request.json', 'P16-trace-B.stop.json',
             'launcher-default-resolution.json', 'stall-gpu.json'):
    path = root/'profile-B'/name
    if path.exists():
        files.append(path)
assert len(set(files)) == len(files)
assert all(p.is_file() and p.stat().st_size < 16*1024*1024 for p in files)
archive = root/'marker-evidence.tar.gz'
with tarfile.open(archive, 'w:gz') as out:
    for path in files:
        out.add(path, arcname=str(path.relative_to(root)))
manifest = dict(archive=archive.name, bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                files=[dict(path=str(p.relative_to(root)), bytes=p.stat().st_size,
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files])
(root/'marker-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps({k:v for k,v in manifest.items() if k!='files'}), 'files', len(files))
