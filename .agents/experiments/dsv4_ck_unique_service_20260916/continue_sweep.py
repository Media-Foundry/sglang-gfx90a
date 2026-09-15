"""After resumed A1 clean shutdown, run the remaining B1/B2/A2 legs."""
import json
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
assert (root/'A1/complete.json').exists()
assert json.loads((root/'A1/P16-ck-unique-A1.stop.json').read_text())['remaining']==[]
for arm in ('B','A2'):
    subprocess.run([sys.executable,str(root/'run_v2.py'),'--arm',arm],check=True)
