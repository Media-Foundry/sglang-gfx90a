"""Paired fixed-token logprobs; top20 is NOT a full-distribution KL oracle."""
import hashlib
import json
from pathlib import Path

import numpy as np

root=Path(__file__).resolve().parent
manifest=json.loads((root/'inputs.json').read_text())
waves={};sources={};hits={}
for arm in ['A1','B','A2']:
    assert json.loads((root/arm/'inputs.json').read_text())==manifest
    assert json.loads((root/arm/'complete.json').read_text())['scored_tokens']==8192
    hits[arm]=json.loads((root/arm/'path-hits.json').read_text())
    for rep in range(2):
        p=root/arm/f'logprobs-{rep}.json';rows=json.loads(p.read_text())
        assert len(rows)==16
        for req,row in zip(manifest['requests'],rows,strict=True):
            assert row['case']==req['index'] and row['token_ids']==req['continuation_ids']
            assert len(row['logprobs'])==len(row['top20'])==256
        waves[f'{arm}.{rep}']=rows
        sources[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()


def stats(a):
    a=np.asarray(a,dtype=np.float64)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),
                p99=float(np.quantile(a,.99)),maximum=float(a.max()))


def compare(left,right):
    a,b=waves[left],waves[right]
    la=np.array([x['logprobs'] for x in a]);lb=np.array([x['logprobs'] for x in b])
    assert np.isfinite(la).all() and np.isfinite(lb).all()
    delta=lb-la
    overlap=[];shared_delta=[];flips=[];exact=0
    for case,(aa,bb) in enumerate(zip(a,b)):
        for position,(ta,tb) in enumerate(zip(aa['top20'],bb['top20'])):
            ma={x[1]:x[0] for x in ta};mb={x[1]:x[0] for x in tb}
            assert len(ma)==len(mb)==20
            common=ma.keys()&mb.keys();overlap.append(len(common)/20)
            shared_delta.extend(abs(ma[k]-mb[k]) for k in common)
            topa=sorted(ta,key=lambda x:(-x[0],x[1]));topb=sorted(tb,key=lambda x:(-x[0],x[1]))
            if topa[0][1]!=topb[0][1]:
                flips.append(dict(case=case,position=position,left_id=topa[0][1],right_id=topb[0][1],
                    left_margin=topa[0][0]-topa[1][0],right_margin=topb[0][0]-topb[1][0],
                    target_logprob_delta=float(delta[case,position])))
            exact+=int(ta==tb)
    return dict(target_logprob_abs_delta=stats(np.abs(delta)),
        mean_nll_left=float(-la.mean()),mean_nll_right=float(-lb.mean()),
        mean_nll_increase=float(-delta.mean()),per_case_nll_increase=(-delta.mean(1)).tolist(),
        target_logprob_exact=int(np.count_nonzero(delta==0)),positions=int(delta.size),
        top20_overlap=stats(overlap),shared_top20_logprob_abs_delta=stats(shared_delta),
        top20_exact_rows=exact,top1_agreement=1-len(flips)/delta.size,top1_flips=flips)


pairs=[('A1.0','A1.1'),('A2.0','A2.1'),('A1.0','A2.0'),('B.0','B.1'),
       ('A1.0','B.0'),('A1.0','B.1'),('A2.0','B.0')]
result=dict(numerical_only=True,identical_teacher_tokens=True,exact_request_echoes=96,
    sources=sources,path_hits=hits,comparisons={f'{a} vs {b}':compare(a,b) for a,b in pairs},
    caveats=['Continuation came from original unshortened-prompt control, not held-out labels.',
             'Top20 overlap is not full-distribution KL or a correctness certificate.',
             'Hit shapes prove executed row sizes, not full internal request ordering.',
             'No wall-time throughput from these logprob-heavy requests.'])
out=root/'analysis.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
for name,r in result['comparisons'].items():
    print(name,'delta_nll',r['mean_nll_increase'],'abs',r['target_logprob_abs_delta'],
          'top1',r['top1_agreement'],'flips',len(r['top1_flips']))
