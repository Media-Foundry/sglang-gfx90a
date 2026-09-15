"""Fresh-process long-output sanity checks after timing; not a speed measurement."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import time
from transformers import AutoTokenizer

root=Path(__file__).resolve().parent;repo=root.parents[2]
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--arm',choices=('A1','B'),required=True)
args=p.parse_args();prior=root/args.arm
assert json.loads((root/'check/validated.json').read_text())['all_eligible_layers_exact']
assert (prior/'complete.json').exists()
out=root/('quality-'+args.arm);out.mkdir(exist_ok=False)
spec=importlib.util.spec_from_file_location('quality_life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
sys.path.insert(0,str(root.parent/'dsv4_ck_unique_service_20260916'))
import stable_lifecycle
stable_lifecycle.install(life);life.ROOT=life.OLD=out
(out/'start-ar-matrix.sh').write_text((prior/'start-ar-matrix.sh').read_text())
plan=json.loads((prior/'plan.json').read_text())
sources=plan['sources']
assert sources=={path:hashlib.sha256((repo/path).read_bytes()).hexdigest() for path in sources}
life.save('plan.json',plan)
manifest=json.loads((prior/'inputs.json').read_text());life.save('inputs.json',manifest)
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
label='P16-h16-quality-'+args.arm;state=life.start(label,0)
try:
    life.ready(state);answers=[]
    for rep in range(2):
        life.resources('quality-'+str(rep),life.owned(state))
        rids=[f'h16-{args.arm}-{rep}-{i}' for i in range(16)]
        payload=dict(input_ids=[r['input_ids'] for r in manifest['requests']],rid=rids,
            cache_salt=[f'h16-quality-{rep}-{i}-{time.time_ns()}' for i in range(16)],
            sampling_params=dict(temperature=0,max_new_tokens=128,ignore_eos=True),return_prompt_token_ids=True)
        result=life.post(life.URL+'/generate',payload,1800);life.save(f'answers-{rep}.json',result)
        byid={r['meta_info']['id']:r for r in result};assert set(byid)==set(rids)
        ordered=[byid[rid] for rid in rids]
        for request,response in zip(manifest['requests'],ordered,strict=True):
            assert response['prompt_token_ids']==request['input_ids'] and response['meta_info']['cached_tokens']==0
            ids=life.completion_ids(response)
            assert len(ids)==128 and tokenizer.decode(ids,skip_special_tokens=False)==response['text']
        answers.append(ordered);print('QUALITY WAVE COMPLETE',args.arm,rep,flush=True)
    logs=Path(state['log']).read_text()
    assert ('H16 peer selected:' in logs)==(args.arm=='B')
    assert 'H16 peer output exact:' not in logs
    assert sources=={path:hashlib.sha256((repo/path).read_bytes()).hexdigest() for path in sources}
    life.save('complete.json',dict(input_echo_exact=32,output_lengths_exact=32,
        quality_repeat_exact=sum(life.completion_ids(a)==life.completion_ids(b) for a,b in zip(*answers,strict=True))))
finally:
    life.stop(state)
