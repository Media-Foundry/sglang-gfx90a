"""Immutable first-layer follow-up archive; never publish tensor dumps."""
import hashlib
import json
from pathlib import Path
import tarfile
from validate_first_layer import main as validate

ROOT=Path(__file__).resolve().parent
RUNS=('combined-l0','combined-ffn','both-ar-l0')


def main():
    validate()
    out=ROOT/'first-layer-evidence.tar.gz'
    manifest=ROOT/'first-layer-evidence-manifest.json'
    assert not out.exists() and not manifest.exists()
    files=[];traces=[]
    for run in RUNS:
        for p in sorted((ROOT/run).rglob('*')):
            if not p.is_file():continue
            if p.suffix=='.pt':
                traces.append(dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
            elif not any(part.startswith('trace-') for part in p.relative_to(ROOT).parts) and p.suffix in ('.json','.log','.sh','.patch'):
                files.append(p)
    for p in sorted(ROOT.iterdir()):
        if p.is_file() and p.suffix in ('.json','.log') and p.name.startswith(('rccl-','woa-service-mutation','combined-l0-run','combined-ffn-run','both-ar-l0-run')):
            files.append(p)
    with tarfile.open(out,'w:gz') as f:
        for p in files:f.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    manifest.write_text(json.dumps(dict(archive_bytes=out.stat().st_size,
        archive_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files],
        local_trace_inventory=traces),indent=2)+'\n')
    print('Archived',len(files),'files,',out.stat().st_size,'bytes; tensor dumps excluded')


if __name__=='__main__':main()
