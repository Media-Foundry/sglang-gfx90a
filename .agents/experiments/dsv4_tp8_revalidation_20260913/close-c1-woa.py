"""Audit C1 ABBA. --accept is used only after manual bounded text review."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys

root = Path(__file__).resolve().parent
repo = Path('/home/pc/Code/sglang')
sys.path.insert(0,str(repo/'scripts/rocm'))
from summarize_dsv4_open_code_matrix import decode_summary

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--accept',action='store_true')
args=parser.parse_args()
arms=[]
for label in ('C1A1','C1B1','C1B2','C1A2'):
    path=root/f'woa-{label}.json'
    data=json.loads(path.read_text())
    assert data['concurrency']==1 and len(data['rounds'])==1
    assert data['requested_decode_seconds_per_round']==30
    assert data['rounds'][0]['decode_seconds']>=30
    summary=decode_summary(data)
    signatures=[dict(wave=w['wave'],index=q['index'],tokens=len(q['output_ids']),sha256=q['sha256'])
                for w in data['rounds'][0]['waves'] for q in w['requests']]
    arms.append(dict(label=label,**summary,signatures=signatures,artifact=path.name,
                     sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
assert len({x['manifest_sha256'] for x in arms})==1
a=statistics.mean(arms[i]['median'] for i in (0,3))
b=statistics.mean(arms[i]['median'] for i in (1,2))
control_return_pct=100*(arms[3]['median']/arms[0]['median']-1)
b_spread_pct=100*abs(arms[1]['median']-arms[2]['median'])/b
gain_pct=100*(b/a-1)
component=json.loads((root/'woa-component.json').read_text())
for row in component['mutations']:
    assert row['grouped_g1']['finite'] and row['grouped_g1']['replay_exact']
    assert row['grouped_g1']['relative_l2_vs_fp32']<.005
for label in ('Final','WoaB','WoaA2'):
    response=json.loads((root/f'woa-{label}-France.json').read_text())
    assert 'paris' in response['text'].lower()
quality=json.loads((root/'woa-candidate-quality.json').read_text())
assert len(quality)==8
for row in quality:
    assert row['response']['text'] and row['response']['meta_info']['completion_tokens']>0
    if row['name']=='France':
        assert 'paris' in row['response']['text'].lower()
if args.accept:
    assert abs(control_return_pct)<2 and b_spread_pct<2 and gain_pct>2
    assert json.loads((root/'ar-c1-gemv/state.json').read_text())['status']=='complete'
report=dict(status='accepted' if args.accept else 'audited_pending_manual_decision',
            arms=arms,control_mean=a,candidate_mean=b,gain_pct=gain_pct,
            control_return_pct=control_return_pct,b_spread_pct=b_spread_pct,
            candidate_repeat_signatures_exact=arms[1]['signatures']==arms[2]['signatures'],
            component=component['summary']['grouped_g1'],
            caveat='Original weights, different floating reduction, not einsum bitwise equivalence. France/short source smokes are not full model numerical or factual proof. C1 supplement is a separate service profile; other-tier HTTP drain rates remain from the main matrix.')
(root/'c1-woa-acceptance.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='arms'},indent=2))
