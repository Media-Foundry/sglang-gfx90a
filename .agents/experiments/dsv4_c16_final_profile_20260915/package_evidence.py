"""Archive completed v2 markers and separately labeled failed first attempt."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
accepted = root/'capture-v2'
assert json.loads((accepted/'complete.json').read_text())['frames'] == 128
assert json.loads((accepted/'analysis.json').read_text())['snapshots'] == 128
assert (accepted/'details-analysis.json').exists()
assert json.loads((accepted/'P16-markers-B.stop.json').read_text())['remaining'] == []
failed = root/'capture'
assert not (failed/'complete.json').exists()
assert json.loads((failed/'P16-markers-B.stop.json').read_text())['remaining'] == []
files = []
for directory, label in ((accepted, 'accepted'), (failed, 'failed-harness')):
    for p in sorted(directory.rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.log', '.sh', '.patch'):
            assert p.stat().st_size < 16*1024*1024
            files.append((p, label+'/'+str(p.relative_to(directory))))
for name in ('capture.log', 'capture-v2.log', 'analysis-v2.log',
             'capture.py', 'analyze.py', 'path_checks.py', 'test_path_checks.py'):
    files.append((root/name, name))
archive = root/'evidence.tar.gz'
assert not archive.exists()
with tarfile.open(archive, 'w:gz') as out:
    for p, name in files:
        out.add(p, arcname=name)
manifest = dict(files=len(files), bytes=archive.stat().st_size,
    sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    members=[dict(path=name, bytes=p.stat().st_size,
        sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in files])
(root/'evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print({k:v for k,v in manifest.items() if k != 'members'})
