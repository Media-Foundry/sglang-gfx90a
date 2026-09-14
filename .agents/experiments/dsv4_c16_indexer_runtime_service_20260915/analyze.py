"""Validate the complete query-reuse ABBA, keeping local and E2E correctness distinct."""
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics

from transformers import AutoTokenizer
from trace_checks import check_shapes

root=Path(__file__).resolve().parent
def read(p):return json.loads((root/p).read_text())
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
arms=('A1','B','A2')
for arm in arms:assert (root/arm/'complete.json').exists(),f'{arm} incomplete'
manifests={arm:read(arm+'/inputs.json') for arm in arms}
assert manifests['A1']==manifests['B']==manifests['A2']
legs=[];quality={};sources={};shapes=[];outputs={}
compile_shapes={}
for arm in arms:
    prefix=arm+'/P16-qrunm-'+arm
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
    assert b'max_total_num_tokens=1048576' in raw
    assert raw.count(b'prefill empty tiles selected:')>=8
    assert raw.count(b'prefill post-fused4 selected:')>=8
    assert raw.count(b'prefill mix-reuse4 selected:')>=8
    group=16
    runtime_m=int(arm=='B')
    assert read(arm+'/plan.json')['query_group_size']==group
    assert read(arm+'/plan.json')['runtime_m']==bool(runtime_m)
    for rank in range(8):
        assert any(f'TP{rank}]' in line and 'prefill query-reuse4 selected' in line
                   and f'query_group={group}' in line and f'runtime_m={runtime_m}' in line
                   for line in raw.decode(errors='replace').splitlines())
    assert b'DSV4 diagnostic stable' not in raw
    compile_shapes[arm]=check_shapes(raw.decode(errors='replace'),runtime_m)
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
            assert abs(r['total_prompt_tokens']/r['prefill_wall_s']-r['aggregate_input_tok_s'])<1e-6
        log=raw[progress['log_start']:progress['log_end']].decode(errors='replace')
        counts=collections.Counter((int(a),int(b)) for a,b in re.findall(
            r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',log))
        assert counts,'Missing timed-forward evidence'
        shapes.append(counts)
        rates=[r['aggregate_input_tok_s'] for r in data['rounds']]
        compiles=[dict(rank=int(rank),kernel=kernel,seconds=float(seconds))
                  for rank,kernel,seconds in re.findall(
                      r"TP(\d+)\] Triton kernel '([^']+)' took ([0-9.]+) s to compile after serving started",log)]
        legs.append(dict(name=name,rates=rates,median=statistics.median(rates),
            serving_compile_events=compiles,all_rounds_compile_warning_free=not compiles,
            median_wave_s=statistics.median(r['prefill_wall_s'] for r in data['rounds']),
            median_request_ttft_s=statistics.median(t for r in data['rounds'] for t in r['request_ttft_s']),
            forwards=[dict(requests=k[0],rows=k[1],count=v) for k,v in sorted(counts.items())]))
    repeats=[]
    for rep in range(2):
        responses=read(arm+'/quality-'+str(rep)+'.json');assert len(responses)==16
        by_id={r['meta_info']['id']:r for r in responses}
        rids=[f'qrunm-{arm}-{rep}-{i}' for i in range(16)];assert set(by_id)==set(rids)
        ids=[]
        for req,rid in zip(manifests[arm]['requests'],rids,strict=True):
            response=by_id[rid];assert response['prompt_token_ids']==req['input_ids']
            meta=response['meta_info'];assert meta['cached_tokens']==0 and meta['completion_tokens']==128
            tokens=response['output_ids'][-meta['completion_tokens']:]
            assert len(tokens)==128 and tokenizer.decode(tokens,skip_special_tokens=False)==response['text']
            ids.append(tokens)
        repeats.append(ids)
    outputs[arm]=repeats
    quality[arm]=dict(input_echo_exact=32,repeat_exact=sum(a==b for a,b in zip(*repeats,strict=True)))
assert sources['A1']==sources['B']==sources['A2']
assert [l['name'] for l in legs]==['A1','B1','B2','A2']
assert all(x==shapes[0] for x in shapes),'Timed admission changed between arms'
rates={arm:statistics.mean(l['median'] for l in legs if l['name'].startswith(arm)) for arm in ('A','B')}
latencies={arm:statistics.mean(l['median_request_ttft_s'] for l in legs if l['name'].startswith(arm)) for arm in ('A','B')}
comparisons={f'{a}-{b}':sum(x==y for x,y in zip(outputs[a][0],outputs[b][0],strict=True))
             for a,b in [('A1','A2'),('A1','B'),('A2','B')]}
divergences={}
def differences(left_wave, right_wave):
    changed=[]
    for case,(left,right) in enumerate(zip(left_wave,right_wave,strict=True)):
        if left==right:continue
        prefix=next(i for i,(x,y) in enumerate(zip(left,right,strict=True)) if x!=y)
        changed.append(dict(case=case,common_prefix_tokens=prefix,
                            left_token=left[prefix],right_token=right[prefix]))
    return changed
for a,b in [('A1','A2'),('A1','B'),('A2','B')]:
    divergences[f'{a}-{b}']=differences(outputs[a][0],outputs[b][0])
within_arm_divergences={a:differences(*outputs[a]) for a in arms}
all_quality_comparisons={}
for a,b in [('A1','A2'),('A1','B'),('A2','B')]:
    for i in range(2):
        for j in range(2):
            changed=differences(outputs[a][i],outputs[b][j])
            all_quality_comparisons[f'{a}.{i}-{b}.{j}']=dict(exact=16-len(changed),divergences=changed)
result=dict(legs=legs,quality=quality,sources=sources,first_quality_wave_cross_arm_exact=comparisons,
    compile_shapes=compile_shapes,
    all_timed_legs_compile_warning_free=all(l['all_rounds_compile_warning_free'] for l in legs),
    first_quality_wave_divergences=divergences,
    within_arm_divergences=within_arm_divergences,
    all_quality_wave_comparisons=all_quality_comparisons,
    control_mean_leg_median=rates['A'],candidate_mean_leg_median=rates['B'],
    throughput_gain_percent=100*(rates['B']/rates['A']-1),
    request_ttft_change_percent=100*(latencies['B']/latencies['A']-1),
    mean_leg_median_request_ttft_s=latencies,identical_timed_forward_shape_counts=True,
    input_sha256=hashlib.sha256((root/'A1/inputs.json').read_bytes()).hexdigest(),
    scope='Original V4 TP8/native AR C16x8K zero-prefix prefill wave throughput,1M KV; medians retain cold-shape rounds when present, inspect compile events before calling these warm results; not proof of whole-model determinism.')
(root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
