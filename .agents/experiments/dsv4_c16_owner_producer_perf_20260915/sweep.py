"""Run three owned fresh services in A1/B1/B2/A2 order; never implicit retry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--resume-from',choices=['A1','B','A2'],default='A1')
args=p.parse_args();arms=['A1','B','A2'];start=arms.index(args.resume_from)
for arm in arms[:start]:
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-producer-perf-{arm}.stop.json').read_text())['remaining']==[]
for arm in arms[start:]:
    assert not (root/arm).exists(),f'Inspect existing {arm}, do not restart it implicitly'
    print('ARM START',arm,flush=True)
    with (root/f'{arm}-run.log').open('x') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-producer-perf-{arm}.stop.json').read_text())['remaining']==[]
    print('ARM COMPLETE',arm,flush=True)
