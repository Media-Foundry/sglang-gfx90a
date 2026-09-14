"""Verify bounded wq_b repair claims without declaring model determinism."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
def read(path):return json.loads((root/path).read_text())
run='layer0-stable-qkv-wqb/'
assert read(run+'complete.json')['diagnostic_only']
identity=read(run+'identity.json')
assert all(x['echo_exact']==16 and x['hashes']==identity[0]['hashes'] for x in identity)
full=read(run+'prepare-summary.json')
assert all(x['changed_elements']==0 for rows in full['comparisons'].values() for x in rows)
data=read(run+'all-rank-summary.json')
for rank in data['per_rank_comparison'].values():
    for stage in ('prepare_qkv_a','prepare_q_before_norm_rope','prepare_kv','q',
                  'attn_core','attn_inverse_rope','wo_a'):
        assert rank[stage]['rows']==rank[stage]['exact_rows'],stage
assert all(v['rows']==v['exact_rows'] for rank in data['control_repeat'].values() for v in rank.values())
assert data['output_comparisons']['A1-A2']['exact']==16
assert data['output_comparisons']['A1-B1']['exact']==13
oracle=read('wqb-service-oracle.json')
assert len([x for x in oracle if x['backend']=='service-mutations'])==8
for row in oracle:
    if row['backend']=='linear':assert row['changed_vs_service']==[0,0]
    if row['backend']=='service':assert row['shifted_changed']==0
    if row['backend']=='service-mutations':
        assert len(row['trials'])==100 and all(x['exact'] for x in row['trials'])
print('PASS: exact input identity, 800 mutations, sampled attention closure, residual wo_b/shared drift not claimed fixed.')
