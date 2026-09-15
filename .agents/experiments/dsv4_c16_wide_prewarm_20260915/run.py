"""Separate processes ensure priming is tested across a fresh process boundary."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
(root/'v2').mkdir(exist_ok=True)
for mode in ('prime','warm','cold'):
    with (root/'v2'/f'{mode}.log').open('x') as log:
        subprocess.run([sys.executable,str(root/'probe.py'),'--mode',mode],
                       stdout=log,stderr=subprocess.STDOUT,check=True)
    print('COMPLETE',mode,flush=True)
