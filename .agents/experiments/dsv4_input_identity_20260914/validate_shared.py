"""Verify bounded shared repair; router logits and full-model tokens are separate."""
import json
import argparse
from pathlib import Path
root=Path(__file__).resolve().parent
def read(p):return json.loads((root/p).read_text())
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-name',default='layer0-shared-wired')
args=parser.parse_args()
run=args.run_name+'/'
assert read(run+'complete.json')['diagnostic_only']
assert (root/run/'INPUT-IDENTITY.service.log').read_text().count('DSV4 diagnostic stable shared selected')==8
identity=read(run+'identity.json');assert len(identity)==4
assert all(x['echo_exact']==16 and x['hashes']==identity[0]['hashes'] for x in identity)
full=read(run+'prepare-summary.json')
assert all(x['changed_elements']==0 for rows in full['comparisons'].values() for x in rows)
data=read(run+'all-rank-summary.json');assert len(data['per_rank_comparison'])==8
for stages in data['per_rank_comparison'].values():
    for name,value in stages.items():
        if name=='ffn_router_logits':continue
        assert value['rows']>0 and value['rows']==value['exact_rows'],(name,value['rows'],value['exact_rows'])
records=read('shared-service-oracle.json')['records']
assert len(records)==40
for row in records:
    if row['mode']=='linear':assert row['service_mismatches']==[0,0]
    if row['mode']=='fixed-both':assert all(v['elements']==0 for v in row['stages'].values())
    if row['mode']=='service-mutations':assert row['passes']==100
print('PASS: exact input echoes, sampled layer0 attention/FFN closure except separately reported router logits, complete retained KV, 800 shared mutations.')
print('Complete output matches:',{k:v['exact'] for k,v in data['output_comparisons'].items()})
