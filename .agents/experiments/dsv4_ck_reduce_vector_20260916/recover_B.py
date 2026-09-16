"""Independently verify finished B evidence after its final log-prefix assertion failed.

Does not run inference or rewrite observations. Emits an explicitly recovered
completion record only after checking every already-written result and runtime
source. Original harness assertion expected '[TPn]' instead of 'timestamp TPn]'.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import re

root=Path(__file__).resolve().parent;repo=root.parents[2];out=root/'B'
target=out/'complete.json';assert not target.exists()
plan=json.loads((out/'plan.json').read_text())
assert plan['candidate'] and not plan['diagnostic']
for name,digest in plan['sources'].items():
    assert hashlib.sha256((repo/name).read_bytes()).hexdigest()==digest,name
assert not json.loads((out/'P16-vec4-reduce-B.stop.json').read_text())['remaining']
log=(out/'P16-vec4-reduce-B.service.log').read_text()
hits=re.findall(r'TP(\d+)\] DSV4 unique CK fixed-order vec4 reducer selected \(1664 blocks\)',log)
assert sorted(map(int,hits))==list(range(8)),hits
for rank in range(8):
    assert f'unique-slot CK selected: rank={rank} ' in log
    assert f'[TP{rank}] H16 peer selected:' in log
assert 'Scheduler hit an exception' not in log
assert 'paris' in json.loads((out/'France.json').read_text())['text'].lower()
info=json.loads((out/'P16-vec4-reduce-B.server-info.json').read_text())
assert info['max_total_tokens']==1048576 and info['chunked_prefill_size']==32768
progress=json.loads((out/'progress.json').read_text())
assert [r['leg'] for r in progress]==['warmup','B1','B2']
for leg in progress:
    data=json.loads((out/(leg['leg']+'.json')).read_text())
    assert all(r['cached_tokens']==[0]*16 and r['completion_lengths']==[1]*16 for r in data['rounds'])
    assert leg['median']==data['median_input_tok_s']
    assert leg['rates']==[r['aggregate_input_tok_s'] for r in data['rounds']]
spec=importlib.util.spec_from_file_location('life',root.parent/'dsv4_tp8_ar_down_consumer_20260914/trial.py')
life=importlib.util.module_from_spec(spec);spec.loader.exec_module(life)
from transformers import AutoTokenizer
tokenizer=AutoTokenizer.from_pretrained('/home/pc/models/modelscope',local_files_only=True)
inputs=json.loads((out/'inputs.json').read_text())['requests'];waves=[]
for rep in range(4):
    data=json.loads((out/f'quality-{rep}.json').read_text())
    byid={r['meta_info']['id']:r for r in data};assert len(byid)==16
    wave=[byid[f'corrected-B-{rep}-{i}'] for i in range(16)]
    for inp,response in zip(inputs,wave,strict=True):
        ids=life.completion_ids(response)
        assert response['prompt_token_ids']==inp['input_ids']
        assert response['meta_info']['cached_tokens']==0 and len(ids)==128
        assert tokenizer.decode(ids,skip_special_tokens=False)==response['text']
    waves.append(wave)
record=dict(progress=progress,all_layers_exact=False,exact_comparisons=0,diagnostic=False,
    france_passed=True,unique_ck=True,quality_repeat_exact=[sum(life.completion_ids(a)==life.completion_ids(b)
    for a,b in zip(waves[0],w,strict=True)) for w in waves[1:]],
    recovered_after_harness_assertion=True,
    recovery_reason="Final rank-log assertion used '[TPn]' but actual prefix contains timestamp before TPn; all 8 ranks verified independently",
    recovery_sources_sha256={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in
        ['progress.json','quality-0.json','quality-1.json','quality-2.json','quality-3.json','P16-vec4-reduce-B.service.log']})
target.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
