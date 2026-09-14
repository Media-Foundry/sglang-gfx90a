"""Archive completed raw evidence and verify every member, without deleting originals."""
import hashlib
import json
from pathlib import Path
import tarfile

import psutil

ROOT=Path(__file__).resolve().parent


def main():
    assert (ROOT/'summary.json').exists() and (ROOT/'C1-summary.json').exists()
    archive=ROOT/'measurement-evidence.tar.gz'
    assert not archive.exists()
    for path in ROOT.glob('*.state.json'):
        state=json.loads(path.read_text())
        try:
            proc=psutil.Process(state['pid'])
            assert proc.create_time()!=state['birth'] or proc.status()==psutil.STATUS_ZOMBIE, state
        except psutil.NoSuchProcess:
            pass
    files=sorted(p for p in ROOT.iterdir() if p.suffix in ('.json','.log','.patch')
                 and p.name!='evidence-index.json')
    entries=[dict(name=p.name,bytes=p.stat().st_size,
                  sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        for p in files: tar.add(p,arcname=p.name,recursive=False)
    with tarfile.open(archive,'r:gz') as tar:
        assert len(tar.getmembers())==len(entries)
        for entry in entries:
            data=tar.extractfile(entry['name']).read()
            assert len(data)==entry['bytes']
            assert hashlib.sha256(data).hexdigest()==entry['sha256']
    report=dict(archive=archive.name,bytes=archive.stat().st_size,
                sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),files=entries)
    (ROOT/'evidence-index.json').write_text(json.dumps(report,indent=2)+'\n')
    print('VERIFIED',len(entries),'files',report['bytes'],'bytes',report['sha256'])


if __name__=='__main__': main()
