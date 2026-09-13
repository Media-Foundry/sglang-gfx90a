"""Persist individual AMD memory snapshots (CLI watch --file overwrites)."""
import json
from pathlib import Path
import subprocess
import sys
import time
import psutil

service = psutil.Process(int(sys.argv[1]))
birth = service.create_time()
with Path(sys.argv[2]).open('a', buffering=1) as out:
    while service.is_running():
        try:
            if service.create_time() != birth:
                break
            sample = json.loads(subprocess.check_output(
                ['amd-smi', 'metric', '--mem-usage', '--json'], timeout=10))
            out.write(json.dumps({'time':time.time(), 'gpus':sample})+'\n')
        except psutil.NoSuchProcess:
            break
        time.sleep(5)
