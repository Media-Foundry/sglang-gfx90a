"""Read-only CPU build-subprocess witnesses for an owned service; no GPU work."""
import json
from pathlib import Path
import sys
import time
import psutil

service = psutil.Process(int(sys.argv[1]))
birth = service.create_time()
seen = set()
path = Path(sys.argv[2])
with path.open('a', buffering=1) as out:
    while service.is_running():
        try:
            if service.create_time() != birth:
                break
            for p in service.children(recursive=True):
                try:
                    identity = (p.pid, p.create_time())
                    if identity in seen:
                        continue
                    name = p.name()
                    if not any(x in name for x in ('clang', 'hipcc', 'ninja', 'gcc', 'g++')):
                        continue
                    seen.add(identity)
                    out.write(json.dumps({'observed':time.time(), 'pid':p.pid,
                        'created':identity[1], 'name':name, 'cwd':p.cwd(),
                        'command':p.cmdline()})+'\n')
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
        except psutil.NoSuchProcess:
            break
        time.sleep(1)
