"""Validate and package complete-input evidence at the first C4 boundary."""
import hashlib
import json
from pathlib import Path
import tarfile
root=Path(__file__).resolve().parent
run=root/'stable-layer2-prepare'
out=root/'stable-layer2-prepare-evidence.tar.gz'
assert (run/'complete.json').exists() and not out.exists()
full=json.loads((run/'prepare-summary.json').read_text())
assert full['layer']==2 and all(not x['changed_elements'] for rows in full['comparisons'].values() for x in rows)
data=json.loads((run/'all-rank-layer2.json').read_text())
assert len(data['per_rank_comparison'])==8
for stages in data['per_rank_comparison'].values():
    for s in ('prepare_qkv_a','prepare_q_before_norm_rope','prepare_kv','q'):
        assert stages[s]['rows']>0 and stages[s]['rows']==stages[s]['exact_rows']
changes=[d for d in data['per_rank_comparison']['0']['attn_core']['details'] if not d['exact']]
assert min(d['position'] for d in changes)==3 and {d['case'] for d in changes}=={15}
files=[p for p in sorted(run.iterdir()) if p.is_file() and p.suffix in ('.json','.log','.sh','.patch')]
files += [root/n for n in ('stable-layer2-prepare-run.log','stable-layer2-full-analysis.log','stable-layer2-stages-analysis.log')]
with tarfile.open(out,'w:gz') as archive:
    for p in files:archive.add(p,arcname=str(p.relative_to(root)))
manifest=dict(files=len(files),bytes=out.stat().st_size,sha256=hashlib.sha256(out.read_bytes()).hexdigest())
(root/'stable-layer2-prepare-evidence-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(manifest)
