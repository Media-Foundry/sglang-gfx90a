#!/usr/bin/env python3
"""Read-only scoped compiler/lock snapshots; no ptrace or GPU profiler."""
import argparse
import json
import os
from pathlib import Path
import time

import psutil


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pid", type=int, required=True)
    p.add_argument("--seconds", type=int, default=45)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    parent = psutil.Process(a.pid)
    assert "sglang.launch_server" in parent.cmdline(), "not a server parent"
    rows = []
    for _ in range(a.seconds):
        children = parent.children(recursive=True)
        ranks, compilers = [], []
        for child in children:
            try:
                cmd = child.cmdline()
                if cmd and os.path.basename(cmd[0]) in ("ninja", "hipcc", "clang++", "clang-23"):
                    compilers.append({"pid": child.pid, "ppid": child.ppid(),
                                      "exe": cmd[0], "cwd": child.cwd()})
                if "scheduler_TP" not in child.name():
                    continue
                wchan = Path(f"/proc/{child.pid}/wchan").read_text().strip()
                locks = []
                if "lock" in wchan:
                    for fd in Path(f"/proc/{child.pid}/fd").iterdir():
                        try:
                            target = os.readlink(fd)
                            if target.endswith("/.lock"):
                                locks.append(target)
                        except OSError:
                            pass
                ranks.append({"pid": child.pid, "wchan": wchan, "locks": locks})
            except (psutil.Error, OSError):
                continue
        rows.append({"time": time.time(), "ranks": ranks, "compilers": compilers})
        a.output.write_text(json.dumps(rows, indent=2) + "\n")
        if compilers or any(r["locks"] for r in ranks):
            print(json.dumps(rows[-1]), flush=True)
        time.sleep(1)
    print(json.dumps({"samples": len(rows),
                      "compile_samples": sum(bool(r["compilers"]) for r in rows)}), flush=True)


if __name__ == "__main__":
    main()
