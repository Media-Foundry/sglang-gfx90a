"""Require successful core-only boundary ablation and package bounded evidence."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path(__file__).resolve().parent
run=root/'stable-layer2-core-fixed'
assert (run/'complete.json').exists()
assert json.loads((run/'INPUT-IDENTITY.stop.json').read_text())['remaining']==[]
summary=json.loads((run/'compressor-summary.json').read_text())['comparisons']
for name, stages in summary.items():
    for stage,v in stages.items():
        if name=='A1-B1' and stage=='index_projection':
            assert v['changed_elements']==4
        else: assert v['changed_elements']==0,(name,stage)
rank=json.loads((run/'all-rank-layer2.json').read_text())
for key in ('per_rank_comparison','control_repeat'):
    assert len(rank[key])==8
    for stages in rank[key].values():
        assert 'attn_out' in stages and 'ffn_out' in stages
        assert all(v['rows']>0 and v['rows']==v['exact_rows'] for v in stages.values())
full=json.loads((run/'prepare-summary.json').read_text())
assert all(r['changed_elements']==0 for rows in full['comparisons'].values() for r in rows)
files=[p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files += [root/n for n in ('stable-layer2-core-fixed-run.log','core-fixed-compressor-analysis.log',
                          'core-fixed-all-ranks.log','core-fixed-full-prepare.log')]
# Supplemental control checks completed after the first capture package.
baseline=root/'stable-layer2-compressor'
files += [baseline/n for n in ('all-rank-layer2.json','prepare-summary.json')]
out=root/'core-fixed-evidence.tar.gz';assert not out.exists()
with tarfile.open(out,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
manifest=dict(files=len(files),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(root/'core-fixed-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
