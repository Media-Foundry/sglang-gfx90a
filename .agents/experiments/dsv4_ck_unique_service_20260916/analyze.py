"""Same-runtime ABBA summary, preserving warmup and quality boundaries."""
import hashlib,json,statistics
from pathlib import Path

root=Path(__file__).resolve().parent;target=root/'summary.json';assert not target.exists()
plans={a:json.loads((root/a/'plan.json').read_text()) for a in ('A1','B','A2')}
common=set.intersection(*(set(p['sources']) for p in plans.values()))
for path in common:assert len({p['sources'][path] for p in plans.values()})==1,path
assert len({p['input_sha256'] for p in plans.values()})==1
legs={};quality={}
for arm in plans:
    complete=json.loads((root/arm/'complete.json').read_text())
    assert complete['input_echo_exact']==32 and complete['unique_hit']==(arm=='B')
    assert json.loads((root/arm/f'P16-ck-unique-{arm}.stop.json').read_text())['remaining']==[]
    for name in (('B1','B2') if arm=='B' else (arm,)):
        data=json.loads((root/arm/(name+'.json')).read_text())
        assert len(data['rounds'])==3
        for r in data['rounds']:
            assert r['input_echo_exact'] and r['cached_tokens']==[0]*16
            assert r['completion_lengths']==[1]*16 and r['total_prompt_tokens']==131069
        legs[name]=dict(median=data['median_input_tok_s'],rates=[r['aggregate_input_tok_s'] for r in data['rounds']])
    quality[arm]=[json.loads((root/arm/f'quality-{i}.json').read_text()) for i in range(2)]
    assert all(len(x)==16 for x in quality[arm])
control=statistics.mean(legs[a]['median'] for a in ('A1','A2'))
candidate=statistics.mean(legs[a]['median'] for a in ('B1','B2'))
tokens=lambda rows:[r['output_ids'] for r in rows]
summary=dict(scope='TP8 C16x8K original V4, original weights,1M pool, native AR; stages1, query producerOFF',
    legs=legs,control_input_tok_s=control,candidate_input_tok_s=candidate,
    change_pct=(candidate/control-1)*100,control_arm_drift_pct=(legs['A2']['median']/legs['A1']['median']-1)*100,
    interpretation='Throughput parity, not a measured speed gain; deterministic stage2 reference maintained',
    repeat_exact={a:sum(x==y for x,y in zip(tokens(v[0]),tokens(v[1]),strict=True)) for a,v in quality.items()},
    candidate_vs_control_exact={a:sum(x==y for x,y in zip(tokens(quality['B'][0]),tokens(quality[a][0]),strict=True)) for a in ('A1','A2')},
    source_checks=dict(common_paths=len(common),all_equal=True),
    caveats=['A1 process resumed after wall-clock birth guard failure, same PID after warmup',
             'Client perf_counter timing unchanged; warmup excluded',
             'One ABBA with3 rounds/leg; not arbitrary batch-independent or cached-decode determinism'])
target.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
