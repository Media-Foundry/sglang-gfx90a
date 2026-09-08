#!/usr/bin/env python3
"""One owned service, separate P/D tests; one warmup plus three measured waves."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import psutil


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pid', type=int, required=True)
    p.add_argument('--tp', type=int, choices=(4, 8), required=True)
    p.add_argument('--mode', choices=('ar', 'dspark'), required=True)
    p.add_argument('--base-url', required=True)
    p.add_argument('--manifests', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--phases', nargs='+', choices=('decode', 'prefill'), default=['decode', 'prefill'])
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    service = psutil.Process(args.pid)
    birth = service.create_time()
    cmd = service.cmdline()
    assert 'sglang.launch_server' in cmd
    assert cmd[cmd.index('--tp-size') + 1] == str(args.tp)
    spec = '--speculative-algorithm' in cmd
    assert spec == (args.mode == 'dspark')
    if spec:
        assert cmd[cmd.index('--speculative-algorithm') + 1].upper() == 'DSPARK'
    state = dict(status='running', pid=args.pid, tp=args.tp, mode=args.mode,
                 measured_rounds=3, warmup_rounds=1, results=[],
                 decode_tokens=2048, prefill_output_tokens=1)
    def save():
        (args.output_dir / 'state.json').write_text(json.dumps(state, indent=2) + '\n')
    try:
        for phase in args.phases:
            for concurrency in (1, 2, 4, 8, 16, 32):
                for kind, rounds in [('warmup', 1), ('measured', 3)]:
                    assert service.is_running() and service.create_time() == birth
                    owned = {service.pid, *[x.pid for x in service.children(recursive=True)]}
                    gpu = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
                    active = {x['process_info']['pid'] for g in gpu for x in g.get('process_list', [])
                              if isinstance(x.get('process_info'), dict)}
                    assert active <= owned, f'external GPU PIDs: {active-owned}'
                    stem = f'{phase}_c{concurrency}_{kind}'
                    out = args.output_dir / f'{stem}.json'
                    script = ('bench_dsv4_open_code_decode.py' if phase == 'decode'
                              else 'bench_dsv4_prefill_diverse_concurrent.py')
                    command = [sys.executable, str(root / script), '--base-url', args.base_url,
                               '--inputs', str(args.manifests / f'{phase}.json'),
                               '--request-count', str(concurrency), '--rounds', str(rounds),
                               '--tokens', '2048' if phase == 'decode' else '1', '--output', str(out)]
                    if phase == 'decode':
                        command += ['--seconds', '0' if kind == 'warmup' else '30']
                    state.update(current=stem, updated=time.time())
                    save()
                    print('START', stem, flush=True)
                    with (args.output_dir / f'{stem}.log').open('w') as log:
                        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
                    assert result.returncode == 0, f'{stem} failed: see log'
                    data = json.loads(out.read_text())
                    if phase == 'decode':
                        assert data['status'] == 'complete' and data['ignore_eos'] is False
                        if args.mode == 'ar':
                            assert all(req['spec_accept_length'] is None for row in data['rounds']
                                       for wave in row['waves'] for req in wave['requests'])
                    for row in data['rounds']:
                        if phase == 'decode':
                            continue
                        reasons = row['finish_reasons']
                        assert all((x.get('type') if isinstance(x, dict) else x) == 'length' for x in reasons)
                        if phase == 'decode' and args.mode == 'ar':
                            assert row['spec_accept_length_mean'] is None
                    metric = data.get('median_decode_tok_s', data.get('median_input_tok_s'))
                    state['results'].append(dict(phase=phase, concurrency=concurrency, kind=kind,
                                                 output=str(out), rate=metric))
                    print('DONE', stem, metric, flush=True)
        state.update(status='complete', current=None)
    except BaseException as exc:
        state.update(status='failed', error=str(exc))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
