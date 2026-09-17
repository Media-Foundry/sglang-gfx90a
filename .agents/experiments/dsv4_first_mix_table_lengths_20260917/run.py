"""One length at a time; fresh services, unchanged sources, exclusive GPUs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--length', choices=('16k', '32k'), required=True)
args = parser.parse_args()
source = root/'service.py'
digest = hashlib.sha256(source.read_bytes()).hexdigest()
for arm in ('A1', 'B', 'A2'):
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    with (root/f'{args.length}-{arm}-driver.log').open('x') as log:
        subprocess.run([sys.executable, str(source), '--length', args.length, '--arm', arm],
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    stopped = root/args.length/arm/f'P{args.length}-first-table-{arm}.stop.json'
    assert not json.loads(stopped.read_text())['remaining']
    print('FINISHED', args.length, arm, flush=True)
