"""Real-code mixed-prefix check. Run only on an owned idle experiment service.

Prime exact input-ID prefixes; never synthesize or truncate the full request.
Report both full-input and newly-computed-token rates with actual cache hits.
This is separate from the zero-prefix benchmark, not a replacement for it.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import sys
import threading
import time


def make_plan(manifest, page_size=256):
    requests=manifest['requests']
    assert len(requests)==16
    assert len({tuple(r['input_ids']) for r in requests})==16
    plan=[]
    for i,r in enumerate(requests):
        ids=r['input_ids']
        prefix=(len(ids)*(i%4)//4)//page_size*page_size
        assert 0<=prefix<len(ids)
        plan.append(dict(case=i,input_ids=ids,prefix_ids=ids[:prefix],prefix_tokens=prefix,
                         input_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()))
    return plan


def verify_response(response, ids, expected_id, max_cached, completion_tokens=1):
    meta=response['meta_info']
    assert meta['id']==expected_id
    assert response['prompt_token_ids']==ids
    assert meta['prompt_tokens']==len(ids)
    cached=meta['cached_tokens']
    assert type(cached) is int and 0<=cached<=max_cached
    assert meta['completion_tokens']==completion_tokens
    assert len(response['output_ids'])==completion_tokens
    return cached


def validate_cache_pattern(cached, planned, expected=None):
    assert len(cached)==len(planned)==16
    assert all(type(c) is int and 0<=c<=p and (c>0 if p else c==0)
               for c,p in zip(cached,planned,strict=True))
    if expected is not None:
        assert cached==expected, 'Actual cache-hit pattern changed; do not combine incomparable timing waves'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--base-url',required=True)
    p.add_argument('--rounds',type=int,default=3)
    p.add_argument('--tokens',type=int,default=1,choices=(1,128),
                   help='128 is a separate bounded-quality run, not a prefill performance leg.')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan-only',action='store_true')
    p.add_argument('--cache-reference',type=Path,
                   help='Completed control result; require the same input manifest and actual cache-hit vector.')
    args=p.parse_args()
    assert args.rounds>0 and not args.output.exists()
    manifest=json.loads(args.inputs.read_text())
    plan=make_plan(manifest)
    result=dict(status='planned',input_sha256=hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
                plan=plan,rounds=[],completion_tokens=args.tokens,scope=__doc__)
    expected_cache=None
    if args.cache_reference:
        reference=json.loads(args.cache_reference.read_text())
        assert reference['status']=='complete' and reference['rounds']
        assert reference['input_sha256']==result['input_sha256']
        expected_cache=reference['rounds'][0]['cached_tokens']
        for row in reference['rounds']:
            validate_cache_pattern(row['cached_tokens'],[r['prefix_tokens'] for r in plan],expected_cache)
        result['cache_reference_sha256']=hashlib.sha256(args.cache_reference.read_bytes()).hexdigest()
    def save():args.output.write_text(json.dumps(result,indent=2)+'\n')
    save()
    if args.plan_only:return
    sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts/rocm'))
    from bench_dsv4_tp8_mhc_fusion_drift_trial import post
    from bench_dsv4_prefill_diverse_concurrent import post_stream
    result['status']='running';save()
    for rep in range(args.rounds):
        nonce=time.time_ns()
        salts=[f'wide-prefix-{nonce}-{i}' for i in range(16)]
        prime=[r for r in plan if r['prefix_tokens']]
        prime_ids=[f'prime-{nonce}-{r["case"]}' for r in prime]
        prime_payload=dict(input_ids=[r['prefix_ids'] for r in prime],rid=prime_ids,
            cache_salt=[salts[r['case']] for r in prime],return_prompt_token_ids=True,
            sampling_params=dict(temperature=0,max_new_tokens=1,ignore_eos=True))
        start=time.perf_counter()
        primed=post(args.base_url.rstrip('/')+'/generate',prime_payload,1800)
        prime_wall=time.perf_counter()-start
        result['active_round']=dict(round=rep,prime_wall_s=prime_wall,prime_responses=primed,
                                    salts=salts,prime_ids=prime_ids)
        save()
        by_id={r['meta_info']['id']:r for r in primed}
        assert len(primed)==len(by_id)==len(prime)==12 and set(by_id)==set(prime_ids)
        for item,rid in zip(prime,prime_ids,strict=True):
            verify_response(by_id[rid],item['prefix_ids'],rid,0)
        barrier=threading.Barrier(17)
        def request(item):
            i=item['case'];rid=f'full-{nonce}-{i}'
            payload=dict(input_ids=item['input_ids'],rid=rid,cache_salt=salts[i],
                return_prompt_token_ids=True,stream=True,
                sampling_params=dict(temperature=0,max_new_tokens=args.tokens,ignore_eos=True,stream_interval=1))
            barrier.wait()
            response,begin,first,end=post_stream(args.base_url.rstrip('/')+'/generate',payload,1800)
            return dict(case=i,rid=rid,begin=begin,first=first,end=end,response=response)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
            futures=[pool.submit(request,item) for item in plan]
            barrier.wait()
            responses=[f.result() for f in futures]
        result['active_round']['responses']=responses
        save()
        for item,response in zip(plan,responses,strict=True):
            # The priming call computes only its input, not its generated token.
            response['cached_tokens']=verify_response(response['response'],item['input_ids'],
                                                      response['rid'],item['prefix_tokens'],args.tokens)
        cached=[r['cached_tokens'] for r in responses]
        # Neither zero hits nor a changing cache workload may masquerade as
        # a kernel speed difference. Raw responses were saved before this gate.
        validate_cache_pattern(cached,[r['prefix_tokens'] for r in plan],expected_cache)
        expected_cache=cached
        wall=max(r['first'] for r in responses)-min(r['begin'] for r in responses)
        total=sum(len(r['input_ids']) for r in plan)
        computed=total-sum(cached)
        record=dict(round=rep,prime_wall_s=prime_wall,prime_responses=primed,
            cached_tokens=cached,planned_prefix_tokens=[r['prefix_tokens'] for r in plan],
            wave_ttft_s=wall,total_input_tokens=total,newly_computed_tokens=computed,
            full_input_tok_s=total/wall,newly_computed_tok_s=computed/wall,responses=responses)
        result['rounds'].append(record)
        del result['active_round']
        save()
        print(json.dumps({k:v for k,v in record.items() if k not in ('prime_responses','responses')}),flush=True)
    result['status']='complete';save()


if __name__=='__main__':main()
