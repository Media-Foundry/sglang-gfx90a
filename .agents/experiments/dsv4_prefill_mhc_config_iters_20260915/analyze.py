"""Validate the full service ABBA; never turn missing output into no drift."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import re
import statistics


def differences(left, right):
    assert len(left)==len(right)
    result=[]
    for case,(a,b) in enumerate(zip(left,right,strict=True)):
        assert len(a)==len(b)==128
        if a!=b:
            k=next(i for i,(x,y) in enumerate(zip(a,b,strict=True)) if x!=y)
            result.append(dict(case=case,common_prefix_tokens=k,left_token=a[k],right_token=b[k]))
    return result


def quality_wave(manifest,responses,arm,rep,tokenizer):
    assert len(responses)==len(manifest['requests'])==16
    by_id={x['meta_info']['id']:x for x in responses}
    expected=[f'mhc20-{arm}-{rep}-{i}' for i in range(16)]
    assert set(by_id)==set(expected) and len(by_id)==16
    result=[]
    for request,rid in zip(manifest['requests'],expected,strict=True):
        row=by_id[rid];meta=row['meta_info']
        assert row['prompt_token_ids']==request['input_ids']
        assert meta['completion_tokens']==128 and meta['cached_tokens']==0
        assert len(row['output_ids'])>=128
        tokens=row['output_ids'][-128:]
        assert tokenizer.decode(tokens,skip_special_tokens=False)==row['text']
        result.append(tokens)
    return result


def compile_events(log):
    return [dict(rank=int(rank),kernel=kernel,seconds=float(seconds))
            for rank,kernel,seconds in re.findall(
                r"TP(\d+)\] Triton kernel '([^']+)' took ([0-9.]+) s to compile after serving started",log)]


def main(root):
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    def read(path):return json.loads((root/path).read_text())
    arms=('A1','B','A2')
    assert all((root/a/'complete.json').exists() for a in arms),'ABBA incomplete'
    manifests={a:read(a+'/inputs.json') for a in arms}
    assert manifests['A1']==manifests['B']==manifests['A2']
    assert sum(len(r['input_ids']) for r in manifests['A1']['requests'])==524286
    sources={};flags={};legs=[];warmups=[];quality={};waves={};timed={};counts=[]
    for arm in arms:
        prefix=f'{arm}/P16-mhc20-{arm}'
        assert read(prefix+'.stop.json')['remaining']==[]
        info=read(prefix+'.server-info.json')
        assert info['tp_size']==8 and info['ep_size']==1
        assert info['speculative_algorithm'] is None and info['enable_mixed_chunk'] is False
        assert info['max_total_tokens']==1048576
        assert info['chunked_prefill_size']==info['max_prefill_tokens']==32768
        assert info['model_path']=='/home/pc/models/modelscope'
        assert 'paris' in read(arm+'/France.json')['text'].lower()
        plan=read(arm+'/plan.json');sources[arm]=plan['sources'];flags[arm]=plan['flags']
        assert flags[arm]['SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS']==str(int(arm=='B'))
        assert flags[arm]['SGLANG_DSV4_GFX90A_MHC_SINKHORN_ITERS']=='8'
        assert flags[arm]['SGLANG_DSV4_C4_PREFILL_QUERY_WIDE']=='1'
        raw=(root/(prefix+'.service.log')).read_bytes();text=raw.decode(errors='replace')
        assert 'Scheduler hit an exception' not in text
        assert 'max_total_num_tokens=1048576' in text
        for rank in range(8):
            ranklines=[s for s in text.splitlines() if f'TP{rank}]' in s]
            assert any('prefill wide-query-reuse selected:' in s and 'C4_capacity=8192' in s and
                       'query_group=16' in s and 'runtime_m=1' in s for s in ranklines)
            assert any('prefill splitk selected:' in s and 'path=fused_tail' in s and
                       'batch=1' in s and 'weight_dtype=torch.float16' in s for s in ranklines)
            assert any('MHC Sinkhorn policy selected: legacy=8 config=20' in s for s in ranklines)==(arm=='B')
        for entry in read(arm+'/progress.json'):
            name=entry['leg'];data=read(arm+'/'+name+'.json')
            warm=name=='warmup';rounds=data['rounds'];assert len(rounds)==(1 if warm else 3)
            for row in rounds:
                assert row['total_prompt_tokens']==524286
                assert row['completion_lengths']==[1]*16 and row['cached_tokens']==[0]*16
                assert len(row['completion_ids'])==len(row['texts'])==16
                for ids,txt in zip(row['completion_ids'],row['texts'],strict=True):
                    assert len(ids)==1 and tokenizer.decode(ids,skip_special_tokens=False)==txt
                assert abs(row['aggregate_input_tok_s']-524286/row['prefill_wall_s'])<1e-6
            segment=raw[entry['log_start']:entry['log_end']].decode(errors='replace')
            counter=collections.Counter((int(a),int(b)) for a,b in re.findall(
                r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',segment))
            assert counter
            record=dict(name=f'{arm}.warmup' if warm else name,
                rates=[r['aggregate_input_tok_s'] for r in rounds],
                median=statistics.median(r['aggregate_input_tok_s'] for r in rounds),
                median_wave_s=statistics.median(r['prefill_wall_s'] for r in rounds),
                median_request_ttft_s=statistics.median(t for r in rounds for t in r['request_ttft_s']),
                compile_events=compile_events(segment),
                scheduler_admissions=[dict(requests=k[0],page_rounded_tokens=k[1],count=v) for k,v in sorted(counter.items())])
            if warm:warmups.append(record)
            else:
                counts.append(counter);legs.append(record)
                timed[name]=[[ids[0] for ids in r['completion_ids']] for r in rounds]
        pair=[]
        for rep in range(2):
            output=quality_wave(manifests[arm],read(f'{arm}/quality-{rep}.json'),arm,rep,tokenizer)
            pair.append(output);waves[f'{arm}.{rep}']=output
        changed=differences(*pair)
        quality[arm]=dict(input_echo_exact=32,repeat_exact=16-len(changed),divergences=changed)
    assert sources['A1']==sources['B']==sources['A2']
    other_flags={a:{k:v for k,v in flags[a].items() if k!='SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS'} for a in arms}
    assert other_flags['A1']==other_flags['B']==other_flags['A2']
    assert [x['name'] for x in legs]==['A1','B1','B2','A2']
    assert all(c==counts[0] for c in counts),'Page-rounded admission counts changed; investigate before comparing'
    rates={a:statistics.mean(x['median'] for x in legs if x['name'].startswith(a)) for a in ('A','B')}
    ttft={a:statistics.mean(x['median_request_ttft_s'] for x in legs if x['name'].startswith(a)) for a in ('A','B')}
    comparisons={}
    for a,b in [('A1','A2'),('A1','B'),('A2','B')]:
        for i in (0,1):
            for j in (0,1):
                changed=differences(waves[f'{a}.{i}'],waves[f'{b}.{j}'])
                comparisons[f'{a}.{i}-{b}.{j}']=dict(exact=16-len(changed),divergences=changed)
    result=dict(legs=legs,warmups=warmups,quality=quality,sources=sources,flags=flags,
        control_mean_leg_median=rates['A'],candidate_mean_leg_median=rates['B'],
        throughput_change_percent=100*(rates['B']/rates['A']-1),
        mean_leg_median_request_ttft_s=ttft,request_ttft_change_percent=100*(ttft['B']/ttft['A']-1),
        formal_compile_warning_free=all(not x['compile_events'] for x in legs),
        all_quality_wave_comparisons=comparisons,timed_first_token_ids=timed,
        identical_scheduler_admission_counts=True,actual_forward_M_equality_proven=False,
        bounded_coherence_review='pending manual review',
        scope='Original V4 TP8 C16x32K, zero-prefix input/wave-time; native AR and1M KV. Only opt-in prefill Sinkhorn config20 differs. Not a determinism/accuracy certificate.')
    (root/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    main(p.parse_args().root)
