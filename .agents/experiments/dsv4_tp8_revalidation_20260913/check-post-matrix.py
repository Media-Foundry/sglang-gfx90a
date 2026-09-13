"""Run only after matrix traffic ends: fresh-cache France + real8K code smoke."""
import argparse
import json
from pathlib import Path
import sys
import time
import subprocess
import psutil

ROOT = Path('/home/pc/Code/sglang')
OUT = ROOT/'.agents/experiments/dsv4_tp8_revalidation_20260913'
sys.path.insert(0, str(ROOT/'scripts/rocm'))
from bench_dsv4_tp8_mhc_fusion_drift_trial import get, post
from bench_dsv4_tp4_diverse_concurrent import completion_ids

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--matrix', type=Path, default=OUT/'ar-matrix')
parser.add_argument('--output', type=Path, default=OUT/'post-matrix-quality.json')
args = parser.parse_args()
assert not args.output.exists(), 'Preserve existing quality evidence'
state = json.loads((args.matrix/'state.json').read_text())
assert state['status'] == 'complete', 'Never contaminate matrix timing'
service = psutil.Process(state['pid'])
assert service.cmdline() == state['command']
birth = service.create_time()
def resources():
    assert service.is_running() and service.create_time() == birth
    owned = {service.pid, *[p.pid for p in service.children(recursive=True)]}
    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    active = {p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
              if isinstance(p.get('process_info'), dict)}
    assert active <= owned, f'Other GPU users: {active-owned}'
    with args.output.with_suffix('.gpu-before.jsonl').open('a') as out:
        out.write(json.dumps(dict(time=time.time(), gpus=gpu))+'\n')
resources()
info = get('http://127.0.0.1:30021/server_info')
assert info['tp_size'] == 8 and info['speculative_algorithm'] is None
france = json.loads((ROOT/'.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests'][0]
corpus = json.loads((OUT/'manifests64/prefill.json').read_text())['requests']
results = []
for repeat in range(2):
    for name, item, limit in [('France',france,32),
                              *[(f'code8k-{i}',corpus[i],128) for i in (0,7,21)]]:
        resources()
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
        args.output.write_text(json.dumps(results,indent=2)+'\n')
        print(repeat,name,repr(response['text']),flush=True)
print('COMPLETE: count/native/cache gates passed; inspect code text semantically.',flush=True)
