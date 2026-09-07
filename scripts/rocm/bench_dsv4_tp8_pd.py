"""Real-request TP8 P/D acceptance suite; does not launch or stop services."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:30011')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--arm', required=True)
    p.add_argument('--reference-dir', type=Path)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    def run(name, script, options):
        path = args.output_dir / f'{name}.json'
        if path.exists():
            raise FileExistsError(path)
        print(args.arm, name, flush=True)
        with (args.output_dir / f'{name}.stdout').open('w') as out:
            subprocess.run([sys.executable, f'scripts/rocm/{script}', '--base-url', args.base_url,
                '--output', str(path), *options], cwd=root, stdout=out, stderr=subprocess.STDOUT,
                check=True, timeout=1200)
    c1 = ['--arm', args.arm, '--rounds', '3']
    if args.reference_dir:
        c1 += ['--reference', str(args.reference_dir / 'decode_c1.json')]
    run('decode_c1', 'bench_dsv4_c1_mhc_recovery.py', c1)
    run('prefill_c1', 'bench_dsv4_prefill_diverse_concurrent.py',
        ['--request-count', '1', '--request-offset', '2', '--tokens', '1', '--rounds', '5'])
    run('prefill_c32', 'bench_dsv4_prefill_diverse_concurrent.py',
        ['--request-count', '32', '--tokens', '1', '--rounds', '3'])
    run('decode_c32', 'bench_dsv4_tp4_diverse_concurrent.py',
        ['--request-count', '32', '--request-seed', '20260907', '--tokens', '256',
         '--rounds', '4', '--require-france-exact', '--resident-time-bins', '4'])
    run('oracle_c32', 'check_dsv4_tp4_m32_next_token.py',
        ['--inputs', '.agents/memory/dsv4_prefill_diverse_32_input_ids.json',
         '--request-count', '32', '--tokens', '32'])
    print(args.arm, 'complete', flush=True)


if __name__ == '__main__':
    main()
