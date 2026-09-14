"""Archive completed arm evidence; verify hashes and keep original files intact."""
import hashlib
import json
from pathlib import Path
import tarfile

import psutil

ROOT = Path(__file__).resolve().parent


def main():
    summary = json.loads((ROOT/'summary.json').read_text())
    archive = ROOT/'measurement-evidence.tar.gz'
    assert not archive.exists()
    for arm in ('A1', 'B', 'A2'):
        state = json.loads((ROOT/arm/f'P16-{arm}.state.json').read_text())
        try:
            proc = psutil.Process(state['pid'])
            assert proc.create_time() != state['birth'] or proc.status() == psutil.STATUS_ZOMBIE
        except psutil.NoSuchProcess:
            pass
    entries = summary['artifact_manifest']
    with tarfile.open(archive, 'w:gz', compresslevel=6) as tar:
        for name, entry in entries.items():
            path = ROOT/name
            assert path.stat().st_size == entry['bytes']
            assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256']
            tar.add(path, arcname=name, recursive=False)
    with tarfile.open(archive, 'r:gz') as tar:
        assert len(tar.getmembers()) == len(entries)
        for name, entry in entries.items():
            data = tar.extractfile(name).read()
            assert len(data) == entry['bytes']
            assert hashlib.sha256(data).hexdigest() == entry['sha256']
    report = dict(archive=archive.name, bytes=archive.stat().st_size,
                  sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), members=len(entries))
    (ROOT/'evidence-index.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
