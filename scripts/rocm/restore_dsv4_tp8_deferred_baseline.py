#!/usr/bin/env python3
"""Restore only the known TP8/1M loopback experiment after its controller exits."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--completed-state', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists(), 'use a new restore state path'
    state = json.loads(a.completed_state.read_text())
    flag = state['candidate_flag']
    assert flag in ('SGLANG_DSV4_GFX90A_TP8_M32_DEFERRED_FINALIZE',
                    'SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM')
    assert state['status'] == 'complete'
    assert state['service_pid'] == a.pid
    service = psutil.Process(a.pid)
    cmd, env, cwd = service.cmdline(), service.environ(), service.cwd()
    assert cmd[1:3] == ['-m', 'sglang.launch_server']
    for key, value in {'--tp-size': '8', '--ep-size': '1', '--port': '30011',
                       '--max-total-tokens': '1048576', '--host': '127.0.0.1',
                       '--model-path': '/home/pc/models/modelscope'}.items():
        assert cmd[cmd.index(key) + 1] == value
    assert env.get(flag) == '1'
    children = service.children(recursive=True)
    owned = {a.pid, *[p.pid for p in children]}
    data = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    seen = {int(x['process_info']['pid']) for g in data for x in g.get('process_list', [])}
    assert not seen - owned, ('foreign GPU processes', sorted(seen - owned))
    record = dict(status='stopping', old_pid=a.pid, removed_flag=flag)
    def save():
        a.output.write_text(json.dumps(record, indent=2) + '\n')
    save()
    service.terminate()
    _, alive = psutil.wait_procs([service, *children], timeout=20)
    for child in alive:
        child.terminate()
    _, alive = psutil.wait_procs(alive, timeout=10)
    assert not alive, [(p.pid, p.status()) for p in alive]
    env.pop(flag)
    log_path = a.output.with_suffix('.service.log')
    with log_path.open('w') as log:
        proc = subprocess.Popen(['numactl', '--interleave=all', *cmd], cwd=cwd,
            env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    record.update(status='starting', pid=proc.pid, service_log=str(log_path))
    save()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        assert proc.poll() is None, ('service exited', proc.returncode)
        try:
            with opener.open('http://127.0.0.1:30011/health', timeout=3) as response:
                if response.status == 200:
                    break
        except Exception:
            pass
        time.sleep(5)
    else:
        record['status'] = 'readiness_observation_timeout'
        save()
        raise TimeoutError(f'Inspect same live PID {proc.pid}; do not restart')
    france = a.output.with_suffix('.france.json')
    with a.output.with_suffix('.france.log').open('w') as log:
        subprocess.run([cmd[0], 'scripts/rocm/check_dsv4_france_c32.py',
                        '--output', str(france)], cwd=cwd, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    result = json.loads(france.read_text())
    assert result['exact_count'] == result['request_count'] == 32
    record.update(status='validated', france_prefix_exact=32, france=str(france))
    save()
    print(json.dumps(record))


if __name__ == '__main__':
    main()
