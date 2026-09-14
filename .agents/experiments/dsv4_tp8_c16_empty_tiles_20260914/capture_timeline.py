"""Separate post-ABBA trace; named to avoid shadowing stdlib profile/cProfile."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import requests

root=Path(__file__).resolve().parent;repo=root.parents[2]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--source-arm',choices=('A2','B'),required=True)
args=p.parse_args()
# Do not overlap profiling with any ABBA arm or assume an unfinished run stopped.
for arm in ('A1','B','A2'):
    assert (root/arm/'complete.json').exists(),f'ABBA {arm} not complete'
    stop=root/arm/f'P16-empty-{arm}.stop.json'
    assert json.loads(stop.read_text())['remaining']==[]
out=root/('profile-'+args.source_arm);out.mkdir(exist_ok=False)
helper=repo/'.agents/experiments/dsv4_tp8_ar_down_consumer_20260914/trial.py'
spec=importlib.util.spec_from_file_location('profile_life',helper)
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
life.ROOT=life.OLD=out
source=root/args.source_arm
launcher=(source/'start-ar-matrix.sh').read_text()
(out/'start-ar-matrix.sh').write_text(launcher)
manifest=json.loads((source/'inputs.json').read_text());life.save('inputs.json',manifest)
traces=out/'traces';traces.mkdir()
life.save('plan.json',dict(source_arm=args.source_arm,diagnostic_only=True,
    launcher_sha256=hashlib.sha256(launcher.encode()).hexdigest(),
    input_sha256=hashlib.sha256((out/'inputs.json').read_bytes()).hexdigest()))
state=life.start('P16-trace-'+args.source_arm,0)
session=requests.Session();session.trust_env=False
profiling=False
try:
    life.ready(state)
    for name in ('warmup','traced'):
        life.resources(name+'-before',life.owned(state))
        if name=='traced':
            config=dict(output_dir=str(traces),activities=['CPU','GPU'],
                        with_stack=False,record_shapes=True,detailed_annotations=True,
                        profile_id='c16-native-'+args.source_arm,merge_profiles=False)
            life.save('profile-request.json',config)
            response=session.post(life.URL+'/start_profile',json=config,timeout=300)
            life.save('profile-start-response.json',dict(status=response.status_code,text=response.text))
            response.raise_for_status();assert response.text.startswith('Start profiling.')
            profiling=True
        rids=[f'trace-{name}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'trace-{name}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=1),return_prompt_token_ids=True)
        life.save(name+'-sent.json',payload)
        response=life.post(life.URL+'/generate',payload,1800)
        life.save(name+'-response.json',response)
        assert len(response)==16
        by_id={r['meta_info']['id']:r for r in response};assert set(by_id)==set(rids)
        for req,rid in zip(manifest['requests'],rids,strict=True):
            r=by_id[rid]
            assert r['prompt_token_ids']==req['input_ids'] and r['meta_info']['cached_tokens']==0
            assert len(life.completion_ids(r))==1
        if profiling:
            response=session.post(life.URL+'/stop_profile',timeout=600)
            life.save('profile-stop-response.json',dict(status=response.status_code,text=response.text))
            response.raise_for_status();assert response.text.startswith('Stop profiling.')
            profiling=False
    files=[dict(path=str(f.relative_to(out)),bytes=f.stat().st_size) for f in sorted(traces.rglob('*')) if f.is_file()]
    assert files,'Profile endpoint returned but no trace files were written'
    life.save('complete.json',dict(files=files,input_echo_exact=32,diagnostic_only=True))
finally:
    # Owned service cleanup is authoritative even if profiler start/stop fails.
    life.stop(state)
