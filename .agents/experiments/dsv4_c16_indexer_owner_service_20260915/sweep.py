"""Only continue from verified complete/owned-stop states; no implicit retry."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--resume-from',choices=('A1','B','A2'),default='A1')
args=p.parse_args()
assert json.loads((root/'check/complete.json').read_text())['owner_hit']
assert json.loads((root/'check/P16-owner-check.stop.json').read_text())['remaining']==[]
arms=('A1','B','A2');start=arms.index(args.resume_from)
for arm in arms[:start]:
    assert (root/arm/'complete.json').exists()
    assert json.loads((root/arm/f'P16-owner-{arm}.stop.json').read_text())['remaining']==[]
for arm in arms[start:]:
    assert not (root/arm).exists()
    print('ARM START',arm,flush=True)
    with (root/(arm+'-run.log')).open('x') as log:
        subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (root/arm/'complete.json').exists()
    print('ARM COMPLETE',arm,flush=True)
