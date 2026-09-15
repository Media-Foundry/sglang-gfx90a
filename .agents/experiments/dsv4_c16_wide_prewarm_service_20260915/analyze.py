"""Live artifact coverage and raw wave timing, not a cold-service A/B."""
import hashlib
import json
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent;out=root/'service'
complete=json.loads((out/'complete.json').read_text())
assert complete['quality_echoes']==64 and complete['first_warm_echoes']==128
assert json.loads((out/'P16-wide-prewarm.stop.json').read_text())['remaining']==[]
plan=json.loads((out/'plan.json').read_text())
assert not plan['producer'] and plan['stage1'] and plan['wide'] and not plan['cold_cache_control']
primed=json.loads((out/'prewarm.json').read_text())
artifacts={r['artifact']:r for r in primed['records']}
log=(out/'P16-wide-prewarm.service.log').read_bytes()
results=[];sources={}
for leg in complete['progress']:
    name=leg['name'];length=int(name.split('k-')[0]);phase=name.split('-')[-1]
    path=out/f'{name}.json';data=json.loads(path.read_text())
    manifest_path=out/f'inputs{length}k.json';manifest=json.loads(manifest_path.read_text())
    assert data['input_manifest_sha256']==hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    total=sum(len(r['input_ids']) for r in manifest['requests'])
    assert total=={16:262141,32:524286}[length]
    assert len(data['rounds'])==(1 if phase=='first' else 3)
    rates=[];times=[]
    for row in data['rounds']:
        assert row['input_echo_exact'] and row['request_count']==16 and row['total_prompt_tokens']==total
        assert row['cached_tokens']==[0]*16 and row['completion_lengths']==[1]*16
        assert len(row['raw_times'])==16
        seconds=max(t['first'] for t in row['raw_times'])-min(t['begin'] for t in row['raw_times'])
        assert seconds>0 and abs(total/seconds-row['aggregate_input_tok_s'])<1e-8
        rates.append(total/seconds);times.append(seconds)
    assert abs(statistics.median(rates)-leg['median'])<1e-8
    segment=log[leg['log_start']:leg['log_end']].decode(errors='replace')
    stage_hits=[tuple(map(int,m)) for m in re.findall(
        r'\[TP(\d+)\] prefill attention-stage1 selected: rows=(\d+) heads=(\d+) check=(\d+)',segment)]
    assert {r[0] for r in stage_hits}==set(range(8)),name
    assert all(8192<=m<=65536 and h==8 and c==0 for _,m,h,c in stage_hits)
    if phase=='first':
        hits=[r for r in leg['hits'] if r['width']==length*256]
        assert {r['rank'] for r in hits}==set(range(8))
        for hit in hits:
            a=artifacts[hit['artifact']]
            assert (hit['width'],hit['pages'],hit['stride'])==(a['width'],a['page_columns'],a['page_stride'])
            assert hit['align']=='0,0,0,0,0' and hit['runtime']==1 and hit['group']==16
    results.append(dict(name=name,rates=rates,median=statistics.median(rates),wave_seconds=times,
        compilation_markers=[line for line in segment.splitlines() if any(s in line.lower() for s in
            ('ninja','building extension','compiling','building module','linking cxx'))],hits=leg['hits']))
    sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
report=dict(scope=__doc__,legs=results,source_hashes=sources,default_wide_promoted=False)
target=root/'analysis.json';assert not target.exists();target.write_text(json.dumps(report,indent=2)+'\n')
for row in results:print(row['name'],row['rates'],'wave_s',row['wave_seconds'],'compile_markers',len(row['compilation_markers']))
