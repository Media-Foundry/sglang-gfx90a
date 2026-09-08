#!/usr/bin/env python3
"""Replace only the owned TP8/1M loopback diagnostic with paired graphs."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import urllib.request

import psutil

FLAG = 'SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PAIRED_GRAPHS'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--restart-paired', action='store_true',
                        help='explicitly replace an existing paired diagnostic after its controller exits')
    args = parser.parse_args()
    assert not args.output.exists()
    service = psutil.Process(args.pid)
    cmd, env, cwd = service.cmdline(), service.environ(), service.cwd()
    assert cmd[1:3] == ['-m', 'sglang.launch_server']
    for key, value in {'--tp-size': '8', '--ep-size': '1', '--port': '30011',
                       '--max-total-tokens': '1048576', '--host': '127.0.0.1',
                       '--model-path': '/home/pc/models/modelscope',
                       '--moe-a2a-backend': 'none'}.items():
        assert cmd[cmd.index(key) + 1] == value
    assert env.get(FLAG, '0') == ('1' if args.restart_paired else '0')
    assert env.get('SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM', '0') == '0'
    assert not any(x.startswith('--speculative-') for x in cmd)
    children = service.children(recursive=True)
    owned = {service.pid, *(p.pid for p in children)}
    raw = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    seen = {int(p['process_info']['pid']) for gpu in raw for p in gpu.get('process_list', [])}
    assert seen and not seen - owned, ('foreign GPU PIDs', sorted(seen - owned))
    record = dict(status='stopping', old_pid=service.pid, audited_gpu_pids=sorted(seen),
                  candidate_flag=FLAG, kv_pool=1048576)
    # Preserve the exact baseline launch contract before stopping anything.
    # Environment may contain credentials: private file, never print/commit it.
    snapshot = args.output.with_suffix('.private-launch.json')
    fd = os.open(snapshot, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as file:
        json.dump(dict(cmd=cmd, env=env, cwd=cwd), file)
    record['private_launch_snapshot'] = str(snapshot)
    def save():
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    save()
    service.terminate()
    _, alive = psutil.wait_procs([service, *children], timeout=20)
    for process in alive:
        process.terminate()
    _, alive = psutil.wait_procs(alive, timeout=10)
    assert not alive, [(p.pid, p.status()) for p in alive]
    env[FLAG] = '1'
    log_path = args.output.with_suffix('.service.log')
    with log_path.open('w') as log:
        proc = subprocess.Popen(['numactl', '--interleave=all', *cmd], cwd=cwd,
                                env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
    record.update(status='starting', pid=proc.pid, service_log=str(log_path))
    save()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        code = proc.poll()
        if code is not None:
            record.update(status='exited', exit_code=code)
            save()
            raise RuntimeError(f'service exited: inspect {log_path}')
        try:
            with opener.open('http://127.0.0.1:30011/health', timeout=3) as response:
                if response.status == 200:
                    record['status'] = 'ready_not_correctness_validated'
                    save()
                    print(json.dumps(record))
                    return
        except Exception:
            pass
        time.sleep(5)
    record['status'] = 'readiness_observation_timeout'
    save()
    raise TimeoutError(f'Inspect same PID {proc.pid}; do not restart on timeout')


if __name__ == '__main__':
    main()
