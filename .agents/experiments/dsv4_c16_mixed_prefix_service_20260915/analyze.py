"""Validate completed mixed-prefix ABBA; no conclusion from missing responses."""
import collections
import importlib.util
import json
import math
from pathlib import Path
import re
import statistics

ROOT=Path(__file__).resolve().parent
def read(p):return json.loads((ROOT/p).read_text())

def validate_wave_timing(wave):
    """Recompute TTFT from raw timestamps, not only from precomputed rates."""
    rows=wave['responses'];assert len(rows)==16
    for row in rows:
        assert all(math.isfinite(row[k]) for k in ('begin','first','end'))
        assert row['begin']<=row['first']<=row['end']
    elapsed=max(r['first'] for r in rows)-min(r['begin'] for r in rows)
    assert elapsed>0 and math.isfinite(wave['prime_wall_s']) and wave['prime_wall_s']>0
    assert abs(wave['wave_ttft_s']-elapsed)<1e-9
    total=wave['total_input_tokens'];computed=wave['newly_computed_tokens']
    assert type(total) is int and type(computed) is int and 0<computed<=total
    assert abs(wave['full_input_tok_s']-total/elapsed)<1e-6
    assert abs(wave['newly_computed_tok_s']-computed/elapsed)<1e-6

def main():
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
    manifests=[read(a+'/inputs.json') for a in ('A1','B','A2')]
    assert manifests[0]==manifests[1]==manifests[2]
    manifest=manifests[0]
    assert sum(len(r['input_ids']) for r in manifest['requests'])==524286
    spec=importlib.util.spec_from_file_location('prefix_client',ROOT.parent/'dsv4_c16_query_wide_20260915/bench_prefix.py')
    client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)
    plan=client.make_plan(manifest);planned=[p['prefix_tokens'] for p in plan]
    expected=None;legs=[];warmups=[];quality={};sources=[];flags=[]
    for arm in ('A1','B','A2'):
        assert (ROOT/arm/'complete.json').exists()
        prefix=arm+'/P16-prefix-'+arm
        assert read(prefix+'.stop.json')['remaining']==[]
        server=read(prefix+'.server-info.json')
        assert server['tp_size']==8 and server['ep_size']==1
        assert server['max_total_tokens']==1048576
        assert server['chunked_prefill_size']==server['max_prefill_tokens']==32768
        assert server['speculative_algorithm'] is None and not server['enable_mixed_chunk']
        assert server['model_path']=='/home/pc/models/modelscope'
        assert 'paris' in read(arm+'/France.json')['text'].lower()
        config=read(arm+'/plan.json');sources.append(config['sources'])
        flag=dict(config['flags']);assert flag.pop('SGLANG_DSV4_C4_PREFILL_QUERY_WIDE')==str(int(arm=='B'))
        assert flag['SGLANG_DSV4_PREFILL_MHC_CONFIG_ITERS']==flag['SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20']=='0'
        flags.append(flag)
        raw=(ROOT/(prefix+'.service.log')).read_bytes()
        text=raw.decode(errors='replace')
        assert 'Scheduler hit an exception' not in text
        hits=[line for line in text.splitlines() if 'prefill wide-query-reuse selected:' in line]
        if arm=='B':
            for rank in range(8):
                assert any(f'TP{rank}]' in line and 'query_group=16' in line and 'runtime_m=1' in line for line in hits)
        else:assert not hits
        for entry in read(arm+'/progress.json'):
            name=entry['leg'];data=read(arm+'/'+name+'.json')
            assert data['status']=='complete'
            assert [p['input_ids'] for p in data['plan']]==[p['input_ids'] for p in plan]
            assert [p['prefix_tokens'] for p in data['plan']]==planned
            tokens=128 if name=='quality' else 1
            assert data['completion_tokens']==tokens
            assert len(data['rounds'])==(1 if name=='warmup' else 2 if name=='quality' else 3)
            answers=[]
            for wave in data['rounds']:
                validate_wave_timing(wave)
                cached=wave['cached_tokens'];client.validate_cache_pattern(cached,planned,expected)
                if expected is None:expected=cached
                assert wave['total_input_tokens']==524286
                assert wave['newly_computed_tokens']==524286-sum(cached)
                assert abs(wave['newly_computed_tok_s']-(524286-sum(cached))/wave['wave_ttft_s'])<1e-6
                assert abs(wave['full_input_tok_s']-524286/wave['wave_ttft_s'])<1e-6
                assert len(wave['responses'])==16 and len(wave['prime_responses'])==12
                primed={int(r['meta_info']['id'].rsplit('-',1)[-1]):r for r in wave['prime_responses']}
                assert set(primed)=={p['case'] for p in plan if p['prefix_tokens']}
                for item in plan:
                    if not item['prefix_tokens']:continue
                    response=primed[item['case']]
                    client.verify_response(response,item['prefix_ids'],response['meta_info']['id'],0)
                    assert tokenizer.decode(response['output_ids'],skip_special_tokens=False)==response['text']
                ordered=[]
                for item,response in zip(plan,wave['responses'],strict=True):
                    assert response['case']==item['case']
                    obj=response['response']
                    client.verify_response(obj,item['input_ids'],response['rid'],item['prefix_tokens'],tokens)
                    assert obj['meta_info']['cached_tokens']==cached[item['case']]
                    assert tokenizer.decode(obj['output_ids'],skip_special_tokens=False)==obj['text']
                    ordered.append(obj['output_ids'])
                answers.append(ordered)
            segment=raw[entry['log_start']:entry['log_end']].decode(errors='replace')
            compiles=re.findall(r"Triton kernel '([^']+)' took ([0-9.]+) s to compile after serving started",segment)
            if name=='quality':
                quality[arm]=dict(input_echo_exact=32,
                    full_repeat=sum(a==b for a,b in zip(*answers,strict=True)),
                    first_repeat=sum(a[0]==b[0] for a,b in zip(*answers,strict=True)),answers=answers)
            else:
                row=dict(leg=arm+'.warmup' if name=='warmup' else name,
                    newly_computed_rates=[r['newly_computed_tok_s'] for r in data['rounds']],
                    full_input_rates=[r['full_input_tok_s'] for r in data['rounds']],
                    median=statistics.median(r['newly_computed_tok_s'] for r in data['rounds']),
                    median_wave_s=statistics.median(r['wave_ttft_s'] for r in data['rounds']),
                    compile_warnings=compiles,
                    page_rounded_admissions=dict(collections.Counter(re.findall(
                        r'Prefill batch, #new-seq: (\d+), #new-token: (\d+)',segment))))
                # JSON keys must be strings; counts are diagnostic, not exact-M proof.
                row['page_rounded_admissions']={f'{k[0]}:{k[1]}':v for k,v in row['page_rounded_admissions'].items()}
                (warmups if name=='warmup' else legs).append(row)
    assert sources[0]==sources[1]==sources[2] and flags[0]==flags[1]==flags[2]
    assert [r['leg'] for r in legs]==['A1','B1','B2','A2']
    a=statistics.mean(r['median'] for r in legs if r['leg'].startswith('A'))
    b=statistics.mean(r['median'] for r in legs if r['leg'].startswith('B'))
    cross={}
    for left,right in [('A1','A2'),('A1','B'),('A2','B')]:
        for i in (0,1):
            for j in (0,1):
                pairs=list(zip(quality[left]['answers'][i],quality[right]['answers'][j],strict=True))
                cross[f'{left}.{i}-{right}.{j}']=dict(full_exact=sum(a==b for a,b in pairs),
                    first_exact=sum(a[0]==b[0] for a,b in pairs))
    result=dict(scope='C16x32K mixed prefix: newly computed input tokens / wave TTFT, priming excluded; not zero-prefix throughput.',
        control=a,candidate=b,gain_percent=100*(b/a-1),actual_cached_tokens=expected,
        newly_computed_tokens=524286-sum(expected),total_input_tokens=524286,
        legs=legs,warmups=warmups,quality=quality,quality_cross=cross,sources=sources[0],
        formal_compile_warning_free=all(not r['compile_warnings'] for r in legs),
        bounded_manual_review_required=True,global_determinism_proven=False)
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','quality')},indent=2))

if __name__=='__main__':main()
