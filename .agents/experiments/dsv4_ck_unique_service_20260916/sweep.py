"""Sequential owned-service ABBA; never start another arm after failure."""
from pathlib import Path
import subprocess
import sys

root=Path(__file__).resolve().parent
for arm in ('A1','B','A2'):
    subprocess.run([sys.executable,str(root/'run.py'),'--arm',arm],check=True)
