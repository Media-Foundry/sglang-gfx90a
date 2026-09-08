#!/usr/bin/env python3
"""Same-process native TP8 M32 graph A/B; never restarts or kills a service."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import urllib.request

import psutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pid', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixed-warmup-single', action='store_true',
                        help='validate one fresh single-graph arm; no runtime arm switching')
    args = parser.parse_args()
    assert not args.output.exists()
    process = psutil.Process(args.pid)
    cmd = process.cmdline()
    for key, value in {'--tp-size': '8', '--ep-size': '1', '--host': '127.0.0.1',
                       '--port': '30011', '--max-total-tokens': '1048576'}.items():
        assert cmd[cmd.index(key) + 1] == value
    env = process.environ()
    if args.fixed_warmup_single:
        assert env.get('SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PAIRED_GRAPHS', '0') == '0'
        assert env.get('SGLANG_DSV4_GFX90A_TP8_M32_DOWN_FIXED_WARMUP') == '1'
        assert env.get('SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM') in ('0', '1')
        arms = [env['SGLANG_DSV4_GFX90A_TP8_M32_DOWN_UNIFORM'] == '1']
    else:
        assert env.get('SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PAIRED_GRAPHS') == '1'
        arms = [True, False, False, True]
    reference = Path('/tmp/dsv4_runtime_m_c1_B_20260908.json')
    ref = {r['case']: r['output_ids'] for r in json.loads(reference.read_text())['measurements']
           if r['rep'] == 0}
    record = dict(status='waiting_ready', pid=args.pid, blocks=[],
                  fixed_warmup_single=args.fixed_warmup_single,
                  attention_issue_order=int(env.get('SGLANG_DSV4_GFX90A_TP4_M32_ATTN_ISSUE_ORDER', '0')),
                  ar_blocks=int(env.get('SGLANG_DSV4_GFX90A_TP8_M32_AR_BLOCKS', '0')))
    def save():
        args.output.write_text(json.dumps(record, indent=2) + '\n')
    def alive():
        assert process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    def audit():
        alive()
        raw = json.loads(subprocess.check_output(['amd-smi', 'process', '--json']))
        seen = {int(p['process_info']['pid']) for g in raw for p in g.get('process_list', [])
                if isinstance(p['process_info'], dict)}
        owned = {process.pid, *(p.pid for p in process.children(recursive=True))}
        assert seen and not seen - owned, ('foreign GPU PIDs', sorted(seen - owned))
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    def arm(value):
        request = urllib.request.Request('http://127.0.0.1:30011/set_internal_state',
            data=json.dumps({'server_args': {'dsv4_down_uniform_arm': int(value)}}).encode(),
            headers={'Content-Type': 'application/json'})
        with opener.open(request, timeout=30) as response:
            result = json.load(response)
        assert result == [True], ('arm rejected', value, result)
    def run(argv, output):
        with output.with_suffix('.log').open('w') as log:
            subprocess.run([cmd[0], *argv, '--output', str(output)],
                           cwd=process.cwd(), stdout=log, stderr=subprocess.STDOUT,
                           check=True)
        return json.loads(output.read_text())
    save()
    try:
        for _ in range(60):
            alive()
            try:
                with opener.open('http://127.0.0.1:30011/health', timeout=3) as response:
                    if response.status == 200:
                        break
            except Exception:
                pass
            time.sleep(5)
        else:
            raise TimeoutError('readiness observation timeout; inspect same PID')
        for index, candidate in enumerate(arms):
            audit()
            if not args.fixed_warmup_single:
                arm(candidate)
            block = dict(index=index, candidate=candidate, status='correctness')
            record['blocks'].append(block)
            record['status'] = 'running'
            save()
            prefix = args.output.with_suffix('').as_posix() + f'.block{index}'
            france = Path(prefix + '.france.json')
            check = run(['scripts/rocm/check_dsv4_france_c32.py'], france)
            assert check['exact_count'] == check['request_count'] == 32
            block['france'] = str(france)
            c1 = Path(prefix + '.c1.json')
            result = run(['scripts/rocm/bench_dsv4_c1_mhc_recovery.py', '--arm', str(index),
                          '--rounds', '4', '--skip-freeze-gc', '--reference', str(reference)], c1)
            measured = [r for r in result['measurements'] if r['rep'] >= 0]
            assert len(measured) == 12 and all(r['output_ids'] == ref[r['case']] for r in measured)
            block.update(c1=str(c1), c1_reference_exact=12, status='c32')
            save()
            audit()
            c32 = Path(prefix + '.c32.json')
            run(['scripts/rocm/bench_dsv4_tp4_diverse_concurrent.py',
                 '--base-url', 'http://127.0.0.1:30011', '--inputs',
                 '/tmp/dsv4_tp8_c32_ar_code_workload_20260908.json',
                 '--request-count', '32', '--tokens', '256', '--rounds', '6',
                 '--save-output-ids'], c32)
            block.update(c32=str(c32), status='complete')
            save()
            label = 'fixed-warmup single-process arm' if args.fixed_warmup_single else 'same-process block'
            print('Completed', label, index, 'candidate', candidate, flush=True)
            time.sleep(2)
        if not args.fixed_warmup_single:
            arm(False)
        record.update(status='complete_pending_c32_hash_review',
                      final_arm=arms[-1] if args.fixed_warmup_single else False)
        save()
    except Exception as error:
        record.update(status='failed', error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()
