"""Trace layer0 at positions where full layer1 inputs actually diverged."""
import json
import argparse
from pathlib import Path
import subprocess
import sys


root = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--stable-qkv', action='store_true')
parser.add_argument('--stable-wqb', action='store_true')
parser.add_argument('--stable-wob', action='store_true')
args = parser.parse_args()
assert not args.stable_wqb or args.stable_qkv
assert not args.stable_wob or args.stable_wqb
source = root / "layer1-prepare" / "prepare-summary.json"
summary = json.loads(source.read_text())
record = next(r for r in summary["comparisons"]["A1-B1"]
              if r["rank"] == 0 and r["stage"] == "prepare_full_input" and r["case"] == 15)
positions = sorted(set(record["changed_rows"]) | {0, 511, 640, 2047, 4095, 8191})
assert len(record["changed_rows"]) == 147
subprocess.run([
    sys.executable, str(root / "run.py"), "--run-name",
    ("layer0-stable-qkv-wqb-wob" if args.stable_wob else
     "layer0-stable-qkv-wqb" if args.stable_wqb else
     "layer0-stable-qkv" if args.stable_qkv else "layer0-changed-rows"),
    "--rank", "-1", "--short", "--fp32-attn-ar", "--stable-woa", "--fp32-ffn-ar",
    "--skip-weight-dumps", "--stage-layer", "0", "--prepare-dump",
    "--sample-positions", ",".join(map(str, positions)),
    *(["--stable-qkv"] if args.stable_qkv else []),
    *(["--stable-wqb"] if args.stable_wqb else []),
    *(["--stable-wob"] if args.stable_wob else []),
], check=True)
