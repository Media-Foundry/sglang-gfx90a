"""Recompute original C16 wave throughput from raw times; no diagnostic timings."""
import hashlib
import json
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent
legs={};sources={};manifests=[];plans=[];formal_echoes=0
for arm in ['A1','B','A2']:
    directory=root/arm
    complete=json.loads((directory/'complete.json').read_text())
    assert complete['diagnostic'] is False and complete['producer_hit']==(arm=='B')
    assert json.loads((directory/f'P16-producer-perf-{arm}.stop.json').read_text())['remaining']==[]
    manifest=json.loads((directory/'inputs.json').read_text());manifests.append(manifest)
    plans.append(json.loads((directory/'plan.json').read_text()))
    log=(directory/f'P16-producer-perf-{arm}.service.log').read_bytes()
    progress={r['leg']:r for r in complete['progress']}
    for leg in (['B1','B2'] if arm=='B' else [arm]):
        path=directory/f'{leg}.json';data=json.loads(path.read_text())
        assert data['input_manifest_sha256']==hashlib.sha256((directory/'inputs.json').read_bytes()).hexdigest()
        assert len(data['rounds'])==3
        rates=[];walls=[];ttfts=[]
        for row in data['rounds']:
            assert row['input_echo_exact'] and row['request_count']==16 and row['total_prompt_tokens']==131069
            assert row['cached_tokens']==[0]*16 and row['completion_lengths']==[1]*16
            assert len(row['raw_times'])==16
            begin=min(t['begin'] for t in row['raw_times']);end=max(t['first'] for t in row['raw_times'])
            assert end>begin
            rate=131069/(end-begin)
            assert abs(rate-row['aggregate_input_tok_s'])<1e-8
            assert abs((end-begin)-row['prefill_wall_s'])<1e-9
            rates.append(rate);walls.append(end-begin);ttfts.extend(row['request_ttft_s']);formal_echoes+=16
        assert abs(statistics.median(rates)-data['median_input_tok_s'])<1e-8
        p=progress[leg];segment=log[p['log_start']:p['log_end']].decode(errors='replace')
        mode='producer' if arm=='B' else 'owner'
        hits=[tuple(map(int,m)) for m in re.findall(r'\[TP(\d+)\] prefill query-'+mode+r' selected: rows=(\d+) local_rows=(\d+) width=(\d+)',segment)]
        assert set(x[0] for x in hits)==set(range(8)),(leg,hits)
        assert all(8192<=m<=65536 and w<=2048 for _,m,_,w in hits)
        legs[leg]=dict(rates=rates,median=statistics.median(rates),wall_median_s=statistics.median(walls),
            request_ttft_median_s=statistics.median(ttfts),path_hits=hits,
            compile_markers=[line for line in segment.splitlines() if
                any(token in line.lower() for token in ('ninja','building extension','compiling','building module','linking cxx'))])
        sources[str(path.relative_to(root))]=hashlib.sha256(path.read_bytes()).hexdigest()
assert manifests[0]==manifests[1]==manifests[2]
assert plans[0]['sources']==plans[1]['sources']==plans[2]['sources']
a=statistics.mean(legs[k]['median'] for k in ['A1','A2'])
b=statistics.mean(legs[k]['median'] for k in ['B1','B2'])
result=dict(formal_echoes=formal_echoes,total_input_per_wave=131069,source_hashes=sources,legs=legs,
            control_center=a,candidate_center=b,gain_percent=100*(b/a-1),
            caveat='Candidate is non-bitwise to old BLAS; see fixed-continuation numerical record. Warmup excluded.')
path=root/'analysis.json';assert not path.exists();path.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['source_hashes','legs']},indent=2))
for k,v in legs.items():print(k,v['rates'],v['median'],'wave_s',v['wall_median_s'],'TTFT',v['request_ttft_median_s'])
