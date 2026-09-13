"""Start an isolated original-V4 TP8 test arm, refusing occupied GPUs/logs."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    parser.add_argument('--enabled', type=int, choices=(0, 1), required=True)
    parser.add_argument('--profile-default', action='store_true', help='Remove explicit flag and verify launcher default')
    parser.add_argument('--woa-gemv', type=int, choices=(0, 1), help='Explicit historical C1-only wo_a candidate')
    args = parser.parse_args()
    assert not args.profile_default or args.enabled == 1
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    root = Path(__file__).resolve().parent
    log = root/f'empty-tiles-{args.label}.service.log'
    state_path = root/f'empty-tiles-{args.label}-service.json'
    assert not log.exists() and not state_path.exists()
    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
    active = {p['process_info']['pid'] for g in gpu for p in g.get('process_list', [])
              if isinstance(p.get('process_info'), dict)}
    assert not active, f'GPUs occupied: {active}'
    (root/f'empty-tiles-{args.label}-start.gpu-before.json').write_text(json.dumps(gpu, indent=2)+'\n')
    session = f'dsv4-empty-tiles-{args.label}-20260914'
    flag = 'SGLANG_DSV4_GFX90A_AR_INDEXER_EMPTY_TILE_SKIP'
    launch = (['env', '-u', flag] if args.profile_default else ['env', f'{flag}={args.enabled}'])
    if args.woa_gemv is not None:
        launch.append(f'SGLANG_DSV4_GFX90A_TP8_BS1_WOA_GEMV={args.woa_gemv}')
    launch += ['bash', str(root/'start-ar-matrix.sh')]
    command = shlex.join(launch)+' >'+shlex.quote(str(log))+' 2>&1'
    subprocess.run(['tmux', 'new-session', '-d', '-s', session, command], check=True)
    pane = int(subprocess.check_output(['tmux', 'display-message', '-p', '-t', session, '#{pane_pid}']))
    state = dict(label=args.label, enabled=args.enabled, tmux=session, pane_pid=pane,
                 profile_default=args.profile_default,
                 woa_gemv=args.woa_gemv,
                 log=str(log), launch=launch, created=time.time(),
                 git_head=subprocess.check_output(['git','-C','/home/pc/Code/sglang','rev-parse','HEAD']).decode().strip())
    state_path.write_text(json.dumps(state, indent=2)+'\n')
    for _ in range(120):
        parent = psutil.Process(pane)
        for proc in [parent, *parent.children(recursive=True)]:
            try:
                cmd = proc.cmdline()
                if 'sglang.launch_server' in cmd and '/home/pc/models/modelscope' in cmd:
                    state.update(pid=proc.pid, birth=proc.create_time(), command=cmd)
                    state_path.write_text(json.dumps(state, indent=2)+'\n')
                    print(json.dumps(state, indent=2), flush=True)
                    return
            except psutil.NoSuchProcess:
                continue
        time.sleep(.5)
    raise RuntimeError(f'No serving process identified; inspect {log} and {session}')


if __name__ == '__main__':
    main()
