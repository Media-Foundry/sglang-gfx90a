"""Pack immutable measurement evidence after the service and watchers stop."""
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

import psutil


def main():
    root = Path(__file__).resolve().parent
    state = json.loads((root / 'ar-matrix/state.json').read_text())
    assert state['status'] == 'complete'
    # Refuse a live original service. A reused unrelated PID is also a safe
    # refusal; inspect it before overriding the archival workflow.
    assert not psutil.pid_exists(state['pid']), 'Stop the owned test service first'
    for arm_path in root.glob('empty-tiles-*-service.json'):
        arm = json.loads(arm_path.read_text())
        if 'pid' in arm:
            assert not psutil.pid_exists(arm['pid']), f'Stop owned arm {arm_path.name} first'
    final_path = root/'ar-final-decode/state.json'
    if final_path.exists():
        assert json.loads(final_path.read_text())['status'] == 'complete'
    for name in ('ar-c1-gemv', 'ar-c1-gemv-scope32'):
        extra = root/name/'state.json'
        if extra.exists():
            assert json.loads(extra.read_text())['status'] == 'complete'
    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    (root/'closing-gpu.json').write_text(json.dumps(dict(time=time.time(),gpus=gpu),indent=2)+'\n')
    excluded = {'evidence-index.json'}
    files = sorted(p for p in root.rglob('*')
                   if p.is_file() and p.suffix in ('.json', '.jsonl', '.log')
                   and p.name not in excluded)
    records = []
    for path in files:
        assert not path.is_symlink()
        content = path.read_bytes()
        records.append(dict(path=str(path.relative_to(root)), bytes=len(content),
                            sha256=hashlib.sha256(content).hexdigest()))
    archive = root / 'measurement-evidence.tar.gz'
    with tarfile.open(archive, 'w:gz') as stream:
        for path in files:
            stream.add(path, arcname=str(path.relative_to(root)), recursive=False)
    # Check the bytes actually archived, not just the earlier filesystem read.
    with tarfile.open(archive, 'r:gz') as stream:
        assert len(stream.getmembers()) == len(records)
        for row in records:
            content = stream.extractfile(row['path']).read()
            assert len(content) == row['bytes']
            assert hashlib.sha256(content).hexdigest() == row['sha256']
    index = dict(archive=archive.name, bytes=archive.stat().st_size,
                 sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=records)
    (root / 'evidence-index.json').write_text(json.dumps(index, indent=2)+'\n')
    print('Verified', len(records), 'files, archive bytes', index['bytes'],
          'sha256', index['sha256'])


if __name__ == '__main__':
    main()
