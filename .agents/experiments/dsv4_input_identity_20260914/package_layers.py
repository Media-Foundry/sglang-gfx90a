"""Package all-layer summaries and input evidence, excluding large tensor dumps."""
import hashlib
import json
from pathlib import Path
import tarfile

ROOT=Path(__file__).resolve().parent


def main():
    run=ROOT/'all-layers';out=ROOT/'all-layer-evidence.tar.gz'
    assert (run/'complete.json').exists() and not out.exists()
    d=json.loads((run/'all-layer-summary.json').read_text())
    assert len(d['layers'])==43 and d['input_identity_exact'] and d['sampled_only']
    assert d['first_boundary_difference']['layer']==1
    assert all(v['exact_rows']==v['rows'] for l in d['layers']
               for stages in l['control'].values() for v in stages.values())
    files=[p for p in sorted(run.rglob('*')) if p.is_file()
           and not any(s.startswith('trace-') for s in p.relative_to(run).parts)
           and p.suffix in ('.json','.log','.sh','.patch')]
    files += [ROOT/n for n in ('all-layers-run.log','all-layers-analysis.log','all-layers-l0-analysis.log')]
    with tarfile.open(out,'w:gz') as f:
        for p in files:f.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    with tarfile.open(out) as f:assert not any(n.endswith('.pt') or '/trace-' in n for n in f.getnames())
    (ROOT/'all-layer-evidence-manifest.json').write_text(json.dumps(dict(
        bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,
                    sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files]),indent=2)+'\n')
    print('Packaged43-layer sampled evidence:',len(files),'files,',out.stat().st_size,'bytes')


if __name__=='__main__':main()
