"""Audit reference contraction schedule and exact supply timing evidence."""
import collections
import hashlib
import json
from pathlib import Path
import re
import tarfile
root=Path(__file__).resolve().parent;target=root/'analysis.json';assert not target.exists()
loc=None;buf=collections.defaultdict(list);parts=collections.defaultdict(list)
dotlines=[32,33,35,36,38,39,41,42,44,45,47,48,50,51,53,54]
for line in (root/'reference.amdgcn').read_text().splitlines():
    if '.loc' in line:
        match=re.search(r'candidate.py:(\d+):',line);loc=int(match[1]) if match else None
    if loc in dotlines and line.strip().startswith('v_'):
        buf[loc].append(line.strip())
        if line.strip().startswith('v_readlane_b32'):
            parts[loc].append(buf[loc]);buf[loc]=[]
schedule={}
for line in dotlines:
    assert len(parts[line])==16
    counts=[collections.Counter(op.split()[0] for op in ops) for ops in parts[line]]
    assert [c['v_fmac_f32_e32'] for c in counts]==[15]*13+[13,13,11]
    assert [c['v_pk_mul_f32'] for c in counts]==[0]*13+[1,1,2]
    schedule[line]=[dict(c) for c in counts]
rows=[]
for file in ['screen.json','wide-screen.json']:
    data=json.loads((root/file).read_text());assert data['status']=='complete'
    for c in data['cases']:
        assert c['mutation10_exact'] and c['permutation_exact'] and c['replay100_exact']
        a,b=c['median_ms']['A'],c['median_ms']['B']
        rows.append(dict(m=c['m'],candidate=c['name'],includes_post=c['includes_post'],reference_ms=a,candidate_ms=b,
            latency_reduction_pct=100*(1-b/a)))
archive=root/'reference-evidence.tar.gz';assert not archive.exists()
with tarfile.open(archive,'w:gz') as tf:
    for name in ['reference.amdgcn','reference.llir','reference.ttgir','reference.json']:
        tf.add(root/name,arcname=name)
result=dict(status='component-only-exact-narrow-gain',rows=rows,reference_contraction_schedule=schedule,
    caveat='Pinned current Triton compiler behavior, not a universal portable summation contract',
    archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),production_changed=False)
target.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],rows=rows,archive_sha256=result['archive_sha256']),indent=2))
