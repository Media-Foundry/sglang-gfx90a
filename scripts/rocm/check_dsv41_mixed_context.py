#!/usr/bin/env python3
"""Inject C4 probes after a long request crosses16K, retaining evidence.

This is an isolated-service correctness test, not a performance benchmark.
It reads new server log lines rather than polling the HTTP endpoint. Both
existing checkers keep their original acceptance rules. It never kills or
restarts the model service. A missing overlap marker is incomplete coverage,
even if all answers happen to pass.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path


PENDING = re.compile(r"#pending-token: (\d+)")
RUNNING = re.compile(r"#running-req: (\d+)")


def pending_tokens(line):
    if "Prefill batch," not in line:
        return None
    match = PENDING.search(line)
    return int(match[1]) if match else None


def prefill_with_running_decode(line):
    match = RUNNING.search(line)
    return pending_tokens(line) is not None and match is not None and int(match[1]) > 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prepared", type=Path, required=True)
    p.add_argument("--server-log", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30101")
    p.add_argument("--trigger-pending", type=int, default=5000)
    p.add_argument("--timeout", type=float, default=1200)
    args = p.parse_args()
    prepared = json.loads(args.prepared.read_text())
    input_tokens = len(prepared["input_ids"])
    if not prepared.get("prepared") or not 22000 <= input_tokens <= 24000:
        p.error("use a frozen22K-24K real-code prompt, leaving pool room for short requests")
    if not 0 < args.trigger_pending <= input_tokens - 16384 or args.timeout <= 0:
        p.error("trigger must follow16K of long prefill and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parent
    summary = {"passed": False, "completed": False, "input_tokens": input_tokens,
               "url": args.url, "server_log": str(args.server_log),
               "trigger_pending": args.trigger_pending, "overlap_markers": []}
    children = []
    files = []
    start = time.monotonic()
    try:
        with args.server_log.open() as log:
            log.seek(0, 2)
            summary["log_begin_offset"] = log.tell()

            def spawn(name, command):
                stream = (args.output_dir / f"{name}.stdout").open("x")
                files.append(stream)
                proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                children.append((name, proc))
                summary[name + "_command"] = command
                return proc

            long = spawn("long", [sys.executable, str(root / "check_dsv41_long_code.py"),
                                  "--prepared", str(args.prepared), "--output",
                                  str(args.output_dir / "long.json"), "--url", args.url,
                                  "--timeout", str(args.timeout)])
            short = None
            lines = []
            while True:
                fresh = log.readlines()
                lines.extend(fresh)
                for line in fresh:
                    pending = pending_tokens(line)
                    if short is None and pending is not None and 0 < pending <= args.trigger_pending:
                        summary["trigger_line"] = line.rstrip()
                        summary["trigger_elapsed_s"] = time.monotonic() - start
                        short = spawn("short", [sys.executable, str(root / "check_dsv41_concurrent_smoke.py"),
                                                "--output-dir", str(args.output_dir / "short"),
                                                "--url", args.url, "--rounds", "2"])
                    if short is not None and prefill_with_running_decode(line):
                        summary["overlap_markers"].append(line.rstrip())
                if long.poll() is not None and (short is None or short.poll() is not None):
                    break
                if time.monotonic() - start > args.timeout + 60:
                    raise TimeoutError("mixed-context client deadline exceeded; server left untouched")
                time.sleep(0.25)
            lines.extend(log.readlines())
            (args.output_dir / "server-window.log").write_text("".join(lines))
            summary["log_end_offset"] = log.tell()
            summary["returncodes"] = {name: child.poll() for name, child in children}
            summary["completed"] = short is not None
            long_report = json.loads((args.output_dir / "long.json").read_text())
            summary["long_passed"] = long_report.get("passed", False)
            summary["long_completed"] = long_report.get("completed", False)
            if short is not None:
                short_report = json.loads((args.output_dir / "short" / "summary.json").read_text())
                summary["short_passed"] = short_report.get("passed", False)
                summary["short_requests"] = short_report.get("requests", 0)
            summary["passed"] = bool(
                summary["long_passed"] and summary.get("short_passed")
                and summary.get("short_requests") == 8 and summary["overlap_markers"]
                and all(code == 0 for code in summary["returncodes"].values())
            )
    except Exception as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for _, child in children:
            if child.poll() is None:
                child.terminate()  # Only our HTTP-checker child, never the model.
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        for stream in files:
            stream.close()
        summary["elapsed_s"] = time.monotonic() - start
        with (args.output_dir / "summary.json").open("x") as out:
            json.dump(summary, out, indent=2)
        print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
