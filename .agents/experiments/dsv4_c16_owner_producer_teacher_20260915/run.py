"""Native TP8 fixed-continuation input logprobs, two fresh-cache waves/process."""
import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import time

parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['A1','B','A2'],required=True)
args=parser.parse_args()
root=Path(__file__).resolve().parent;repo=root.parents[2]
out=root/args.arm;out.mkdir(exist_ok=False)
manifest=json.loads((root/'inputs.json').read_text())
source=root.parent/'dsv4_c16_owner_producer_service_20260915/A-quality'
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('teacher_lifecycle',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life);life.ROOT=life.OLD=out
candidate=args.arm=='B'
launcher=(source/'start-ar-matrix.sh').read_text()
needle='exec bash scripts/rocm_dsv4_flash.sh serve';assert launcher.count(needle)==1
flags=(f'export SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER={int(candidate)}\n'
       'export SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK=0\n')
(out/'start-ar-matrix.sh').write_text(launcher.replace(needle,flags+needle))
life.save('inputs.json',manifest)
paths=set(json.loads((source/'plan.json').read_text())['sources'])|{str(Path(__file__).relative_to(repo))}
hashes={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in sorted(paths)}
life.save('plan.json',dict(numerical_only=True,arm=args.arm,sources=hashes,
    input_sha256=hashlib.sha256((root/'inputs.json').read_bytes()).hexdigest()))
label='P16-teacher-'+args.arm;state=life.start(label,0)
try:
    life.ready(state)
    env=life.owned(state).environ()
    assert env['SGLANG_DSV4_C4_PREFILL_QUERY_PRODUCER']==str(int(candidate))
    assert env['SGLANG_DSV4_DEBUG_OWNER_PRODUCER_CHECK']=='0'
    info=json.loads((out/(label+'.server-info.json')).read_text())
    assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==info['max_prefill_tokens']==32768
    assert info['ep_size']==1 and info['speculative_algorithm'] is None
    hits=[]
    for rep in range(2):
        rids=[f'teacher-{args.arm}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'teacher-{args.arm}-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True,
            return_logprob=True,logprob_start_len=[r['logprob_start_len'] for r in manifest['requests']],
            top_logprobs_num=20,return_text_in_logprobs=False)
        life.save(f'sent-{rep}.json',payload)
        begin=Path(state['log']).stat().st_size
        result=life.post(life.URL+'/generate',payload,2400);life.save(f'raw-{rep}.json',result)
        byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
        clean=[]
        for req,rid in zip(manifest['requests'],rids,strict=True):
            response=byid[rid];meta=response['meta_info']
            assert response['prompt_token_ids']==req['input_ids'] and meta['cached_tokens']==0
            assert len(life.completion_ids(response))==1
            lp=meta['input_token_logprobs'];top=meta['input_top_logprobs']
            assert len(lp)==len(top)==257,(rid,len(lp),len(top))
            assert [v[1] for v in lp]==req['input_ids'][req['logprob_start_len']:],rid
            assert lp[0][0] is None and not top[0],rid
            lp=lp[1:];top=top[1:]
            assert all(v[0] is not None and math.isfinite(v[0]) for v in lp),rid
            assert all(len(v)==20 and all(math.isfinite(x[0]) for x in v) for v in top),rid
            assert [v[1] for v in lp]==req['continuation_ids'],rid
            clean.append(dict(case=req['index'],token_ids=[v[1] for v in lp],
                              logprobs=[v[0] for v in lp],top20=top))
        life.save(f'logprobs-{rep}.json',clean)
        suffix=Path(state['log']).read_bytes()[begin:].decode(errors='replace')
        pattern=r'\[TP(\d+)\] prefill query-'+('producer' if candidate else 'owner')+r' selected: rows=(\d+) local_rows=(\d+) width=(\d+)'
        found=[tuple(map(int,m)) for m in re.findall(pattern,suffix)]
        assert set(r for r,*_ in found)==set(range(8)),found
        assert all(8192<=m<=65536 and width<=2048 for _,m,_,width in found),found
        hits.append(found);life.save('path-hits.json',hits)
        assert hashes=={p:hashlib.sha256((repo/p).read_bytes()).hexdigest() for p in hashes}
        print('TEACHER WAVE COMPLETE',rep,'rows',len(clean)*256,'hits',len(found),flush=True)
    life.save('complete.json',dict(input_echo_exact=32,scored_tokens=8192,numerical_only=True))
finally:
    life.stop(state)
