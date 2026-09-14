"""Close prefill ABBA only after all arms, input/quality and ownership checks."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

from transformers import AutoTokenizer

root=Path(__file__).resolve().parent
def read(p):return json.loads((root/p).read_text())
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
arms=('A1','B','A2')
for arm in arms:
    assert (root/arm/'complete.json').exists(),f'{arm} is not complete'
manifests={arm:read(arm+'/inputs.json') for arm in arms}
assert manifests['A1']==manifests['B']==manifests['A2']
legs=[];quality={};sources={};shape_tables=[]
for arm in arms:
    prefix=arm+'/P16-empty-'+arm
    assert read(prefix+'.stop.json')['remaining']==[]
    info=read(prefix+'.server-info.json')
    assert info['tp_size']==8 and info['ep_size']==1 and info['speculative_algorithm'] is None
    assert info['max_total_tokens']==1048576
    assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['model_path']=='/home/pc/models/modelscope'
    assert 'paris' in read(arm+'/France.json')['text'].lower()
    sources[arm]=read(arm+'/plan.json')['sources']
    raw=(root/(prefix+'.service.log')).read_bytes()
    assert b'Scheduler hit an exception' not in raw
    assert (raw.count(b'prefill empty tiles selected:')>=8) if arm=='B' else (
        b'prefill empty tiles selected:' not in raw)
    assert b'DSV4 diagnostic stable' not in raw
    for progress in read(arm+'/progress.json'):
        if progress['leg']=='warmup':continue
        name=progress['leg'];data=read(arm+'/'+name+'.json')
        assert len(data['rounds'])==3
        for r in data['rounds']:
            assert r['total_prompt_tokens']==131069 and r['completion_lengths']==[1]*16
            assert r['cached_tokens']==[0]*16
            assert len(r['completion_ids'])==len(r['texts'])==16
            for ids,text in zip(r['completion_ids'],r['texts'],strict=True):
                assert len(ids)==1 and tokenizer.decode(ids,skip_special_tokens=False)==text
        log=raw[progress['log_start']:progress['log_end']].decode(errors='replace')
        counts=collections.Counter((int(a),int(b)) for a,b in re.findall(
            r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',log))
        assert counts,'Missing timed-forward evidence'
        shape_tables.append(counts)
        rates=[r['aggregate_input_tok_s'] for r in data['rounds']]
        assert all(abs(r['total_prompt_tokens']/r['prefill_wall_s']-r['aggregate_input_tok_s'])<1e-6
                   for r in data['rounds'])
        legs.append(dict(name=name,rates=rates,median=statistics.median(rates),
            median_wave_s=statistics.median(r['prefill_wall_s'] for r in data['rounds']),
            median_request_ttft_s=statistics.median(t for r in data['rounds'] for t in r['request_ttft_s']),
            forwards=[dict(requests=k[0],rows=k[1],count=v) for k,v in sorted(counts.items())]))
    repeats=[]
    for rep in range(2):
        responses=read(arm+'/quality-'+str(rep)+'.json');assert len(responses)==16
        by_id={r['meta_info']['id']:r for r in responses}
        rids=[f'empty-{arm}-{rep}-{i}' for i in range(16)]
        assert set(by_id)==set(rids)
        ids=[]
        for req,rid in zip(manifests[arm]['requests'],rids,strict=True):
            response=by_id[rid]
            assert response['prompt_token_ids']==req['input_ids']
            meta=response['meta_info'];assert meta['cached_tokens']==0 and meta['completion_tokens']==128
            tokens=response['output_ids'][-meta['completion_tokens']:]
            assert len(tokens)==128 and tokenizer.decode(tokens,skip_special_tokens=False)==response['text']
            ids.append(tokens)
        repeats.append(ids)
    quality[arm]=dict(input_echo_exact=32,repeat_exact=sum(a==b for a,b in zip(*repeats,strict=True)))
assert sources['A1']==sources['B']==sources['A2']
assert [l['name'] for l in legs]==['A1','B1','B2','A2']
rates={arm:statistics.mean(l['median'] for l in legs if l['name'].startswith(arm)) for arm in ('A','B')}
latencies={arm:statistics.mean(l['median_request_ttft_s'] for l in legs if l['name'].startswith(arm)) for arm in ('A','B')}
result=dict(legs=legs,quality=quality,sources=sources,
    control_mean_leg_median=rates['A'],candidate_mean_leg_median=rates['B'],
    throughput_gain_percent=100*(rates['B']/rates['A']-1),
    request_ttft_change_percent=100*(latencies['B']/latencies['A']-1),
    mean_leg_median_request_ttft_s=latencies,
    identical_timed_forward_shape_counts=all(x==shape_tables[0] for x in shape_tables),
    input_sha256=hashlib.sha256((root/'A1/inputs.json').read_bytes()).hexdigest(),
    scope='Original V4 TP8/native AR C16x8K zero-prefix prefill wave throughput,1M KV; not whole-model determinism.')
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
