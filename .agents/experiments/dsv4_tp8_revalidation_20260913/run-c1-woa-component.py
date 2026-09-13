"""Existing C1 GEMV oracle, only after matrix traffic ends, on physical GPU0."""
import json
import os
from pathlib import Path
import subprocess
import sys

import psutil

root = Path(__file__).resolve().parent
repo = Path('/home/pc/Code/sglang')
assert json.loads((root/'ar-final-decode/state.json').read_text())['status'] == 'complete'
state = json.loads((root/'empty-tiles-Final-service.json').read_text())
proc = psutil.Process(state['pid'])
assert proc.create_time() == state['birth'] and proc.cmdline() == state['command']
owned = {proc.pid, *[p.pid for p in proc.children(recursive=True)]}
gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
active = {p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
          if isinstance(p.get('process_info'), dict)}
assert active <= owned, active-owned
output = root/'woa-component.json'
assert not output.exists()
(root/'woa-component.gpu-before.json').write_text(json.dumps(gpu,indent=2)+'\n')
env = os.environ.copy()
# Make the physical mapping explicit rather than inheriting a prior task's mask.
for key in ('CUDA_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'GPU_DEVICE_ORDINAL'):
    env.pop(key, None)
env['HIP_VISIBLE_DEVICES'] = '0'
env['PYTHONPATH'] = ':'.join(str(repo/p) for p in (
    'python', 'python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312',
    'python/sglang/kernels/aot/python'))
with (root/'woa-component.log').open('w') as log:
    subprocess.run([sys.executable,str(repo/'scripts/rocm/bench_dsv4_tp8_bs1_woa.py'),
                    '--output',str(output)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
result = json.loads(output.read_text())
assert len(result['mutations']) == 100
for row in result['mutations']:
    for name in ('linear','grouped_g1'):
        assert row[name]['finite'] and row[name]['replay_exact']
        assert row[name]['relative_l2_vs_fp32'] < .005
print(json.dumps({k:result[k] for k in ('median_us','summary')},indent=2))
print('PASS finite/stable/error gates; NOT einsum bitwise equivalence.',flush=True)
