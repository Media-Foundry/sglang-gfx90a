"""Close the completed ABBA using per-leg rates, latency and input witnesses."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

root=Path(__file__).resolve().parent
def read(path):return json.loads((root/path).read_text())
for arm in ('A1','B','A2'):
    assert (root/arm/'complete.json').exists(),f'{arm} not complete'
manifests={arm:read(arm+'/inputs.json') for arm in ('A1','B','A2')}
assert manifests['A1']==manifests['B']==manifests['A2']
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)

legs=[];quality={};sources={}
for arm in ('A1','B','A2'):
    info=read(arm+'/P16-'+arm+'.server-info.json')
    budget=65536 if arm=='B' else 32768
    assert info['tp_size']==8 and info['ep_size']==1 and info['speculative_algorithm'] is None
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==budget
    assert info['max_prefill_tokens']==budget and info['model_path']=='/home/pc/models/modelscope'
    assert 'paris' in read(arm+'/France.json')['text'].lower()
    sources[arm]=read(arm+'/plan.json')['helper_sha256']
    raw=(root/arm/('P16-'+arm+'.service.log')).read_bytes()
    assert b'Scheduler hit an exception' not in raw
    for progress in read(arm+'/progress.json'):
        if progress['leg']=='warmup':continue
        name=progress['leg'];data=read(arm+'/'+name+'.json')
        assert len(data['rounds'])==3
        for r in data['rounds']:
            assert r['total_prompt_tokens']==131069 and r['completion_lengths']==[1]*16
            assert r['cached_tokens']==[0]*16
            for ids,text in zip(r['completion_ids'],r['texts'],strict=True):
                assert tokenizer.decode(ids,skip_special_tokens=False)==text
        log=raw[progress['log_start']:progress['log_end']].decode(errors='replace')
        shapes=collections.Counter((int(a),int(b)) for a,b in re.findall(
            r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',log))
        rates=[r['aggregate_input_tok_s'] for r in data['rounds']]
        legs.append(dict(name=name,budget=budget,rates=rates,median=statistics.median(rates),
            median_wave_s=statistics.median(r['prefill_wall_s'] for r in data['rounds']),
            median_request_ttft_s=statistics.median(t for r in data['rounds'] for t in r['request_ttft_s']),
            forward_counts=[dict(requests=k[0],rows=k[1],count=v) for k,v in sorted(shapes.items())],
            forwards=sum(shapes.values())))
    rows=[read(arm+'/quality-'+str(i)+'.json') for i in range(2)]
    ids=[]
    for responses in rows:
        assert len(responses)==16
        completion=[]
        for req,response in zip(manifests[arm]['requests'],responses,strict=True):
            assert response['prompt_token_ids']==req['input_ids']
            assert response['meta_info']['cached_tokens']==0
            out=response['output_ids'];assert len(out)==128
            assert tokenizer.decode(out,skip_special_tokens=False)==response['text']
            completion.append(out)
        ids.append(completion)
    quality[arm]=dict(input_echo_exact=32,repeat_exact=sum(a==b for a,b in zip(*ids,strict=True)))
assert len(set(sources.values()))==1, sources
assert [leg['name'] for leg in legs]==['A1','B1','B2','A2']
a=statistics.mean(leg['median'] for leg in legs if leg['name'].startswith('A'))
b=statistics.mean(leg['median'] for leg in legs if leg['name'].startswith('B'))
latency={arm:statistics.mean(leg['median_request_ttft_s'] for leg in legs
                            if leg['name'].startswith(arm)) for arm in ('A','B')}
result=dict(legs=legs,control_mean_leg_median=a,candidate_mean_leg_median=b,
            throughput_gain_percent=100*(b/a-1),quality=quality,helper_sources=sources,
            mean_leg_median_request_ttft_s=latency,
            request_ttft_change_percent=100*(latency['B']/latency['A']-1),
            input_sha256=hashlib.sha256((root/'A1/inputs.json').read_bytes()).hexdigest(),
            scope='Original TP8/native AR/C16 x8K batch TTFT throughput, 1M KV; not a whole-model bitwise proof.')
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
