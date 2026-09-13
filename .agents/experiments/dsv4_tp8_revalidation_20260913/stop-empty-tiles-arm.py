"""Gracefully stop only the exact test service recorded by the arm launcher."""
import argparse
import json
from pathlib import Path
import re
import signal
import time

import psutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('label')
    args = parser.parse_args()
    assert re.fullmatch(r'[A-Za-z0-9_-]+', args.label)
    root = Path(__file__).resolve().parent
    state = json.loads((root/f'empty-tiles-{args.label}-service.json').read_text())
    proc = psutil.Process(state['pid'])
    assert proc.create_time() == state['birth']
    assert proc.cmdline() == state['command']
    children = proc.children(recursive=True)
    result = dict(pid=proc.pid, birth=state['birth'], signal='SIGINT', time=time.time(),
                  children=[p.pid for p in children])
    proc.send_signal(signal.SIGINT)
    _, alive = psutil.wait_procs([proc, *children], timeout=20)
    result['remaining'] = [p.pid for p in alive if p.is_running() and p.status() != psutil.STATUS_ZOMBIE]
    (root/f'empty-tiles-{args.label}-stop.json').write_text(json.dumps(result, indent=2)+'\n')
    print('stopped', proc.pid, 'remaining', result['remaining'])
    assert not result['remaining'], 'Inspect survivors before launching another arm'


if __name__ == '__main__':
    main()
