"""Verify sampled wo_b repair without requiring unrelated final-token parity."""
import json
from pathlib import Path

root=Path(__file__).resolve().parent
def read(p):return json.loads((root/p).read_text())
run='layer0-stable-qkv-wqb-wob/'
assert read(run+'complete.json')['diagnostic_only']
identity=read(run+'identity.json')
assert len(identity)==4
assert all(x['echo_exact']==16 and x['hashes']==identity[0]['hashes'] for x in identity)
full=read(run+'prepare-summary.json')
assert all(x['changed_elements']==0 for rows in full['comparisons'].values() for x in rows)
data=read(run+'all-rank-summary.json')
assert len(data['per_rank_comparison'])==8
stages=('prepare_qkv_a','prepare_q_before_norm_rope','prepare_kv','q',
        'attn_core','attn_inverse_rope','wo_a','wo_b_partial','wo_b')
for rank in data['per_rank_comparison'].values():
    for stage in stages:
        assert rank[stage]['rows']>0
        assert rank[stage]['rows']==rank[stage]['exact_rows'],stage
for file in ('wob-oracle.json','wob-service-oracle.json'):
    records=read(file)['records']
    assert len(records)==16
    for row in records:
        if row['backend']=='linear':assert row['service_mismatches']==[0,0]
        else:
            assert row['backend']=='fixed128'
            assert row['shifted_elements']==0 and row['row_mutation_passes']==100
print('PASS: all input echoes, sampled attention including wo_b, full retained KV, and 800 mutations per entry. Whole-model drift remains independently assessed.')
print('Final output comparisons:',{k:v['exact'] for k,v in data['output_comparisons'].items()})
