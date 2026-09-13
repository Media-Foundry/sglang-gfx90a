"""Run only after matrix traffic ends: cache-reuse France + real8K code smoke."""
import json
from pathlib import Path
import sys
import time

ROOT = Path('/home/pc/Code/sglang')
OUT = ROOT/'.agents/experiments/dsv4_tp8_revalidation_20260913'
sys.path.insert(0, str(ROOT/'scripts/rocm'))
from bench_dsv4_tp8_mhc_fusion_drift_trial import get, post
from bench_dsv4_tp4_diverse_concurrent import completion_ids

state = json.loads((OUT/'ar-matrix/state.json').read_text())
assert state['status'] == 'complete', 'Never contaminate matrix timing'
info = get('http://127.0.0.1:30021/server_info')
assert info['tp_size'] == 8 and info['speculative_algorithm'] is None
france = json.loads((ROOT/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
corpus = json.loads((OUT/'manifests64/prefill.json').read_text())['requests']
results = []
for repeat in range(2):
    for name, item, limit in [('France',france,32),
                              *[(f'code8k-{i}',corpus[i],128) for i in (0,7,21)]]:
        payload = dict(input_ids=item['input_ids'],
                       cache_salt=f'post-matrix-{repeat}-{name}-{time.time_ns()}',
                       sampling_params=dict(temperature=0,max_new_tokens=limit,ignore_eos=False))
        response = post('http://127.0.0.1:30021/generate',payload,600)
        ids = completion_ids(response)
        meta = response['meta_info']
        assert ids and len(ids) == meta['completion_tokens']
        assert meta.get('spec_accept_length') is None
        assert meta.get('cached_tokens',0) == 0
        if name == 'France':
            assert 'paris' in response['text'].lower()
        results.append(dict(repeat=repeat,name=name,input_tokens=len(item['input_ids']),response=response))
        (OUT/'post-matrix-quality.json').write_text(json.dumps(results,indent=2)+'\n')
        print(repeat,name,repr(response['text']),flush=True)
print('COMPLETE: count/native/cache gates passed; inspect code text semantically.',flush=True)
