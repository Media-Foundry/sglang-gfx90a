"""Fresh-service TP4 ABBA: legacy Top-K versus deterministic Top-K.

Legacy arms are a speed-only diagnostic, NOT a correct deployment candidate.
Both arms retain the cache-layout and MFMA wave-shuffle correctness fixes.
Only processes launched by this script are stopped; the port must be free.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=30011)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if (args.output_dir / "config.json").exists():
        raise FileExistsError("Use a new output directory; do not overwrite an ABBA record")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        if sock.connect_ex(("127.0.0.1", args.port)) == 0:
            raise RuntimeError("Refusing to replace an existing service on the port")
    env = os.environ.copy()
    env.update(HIP_VISIBLE_DEVICES="4,5,6,7", TP_SIZE="4", EP_SIZE="1",
               MOE_A2A_BACKEND="none", PORT=str(args.port), HOST="127.0.0.1",
               MEM_FRACTION_STATIC="0.80", MAX_TOTAL_TOKENS="65536",
               CHUNKED_PREFILL_SIZE="2304", CUDA_GRAPH_BS_DECODE="1",
               CUDA_GRAPH_MAX_BS_DECODE="1", DISABLE_OVERLAP_SCHEDULE="1",
               SGLANG_DSV4_GFX90A_FP4_MFMA32_PREFILL="1",
               SGLANG_DSV4_GFX90A_FP4_MFMA64_PREFILL="1")
    (args.output_dir / "config.json").write_text(json.dumps({
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "settings": {k: env[k] for k in (
            "HIP_VISIBLE_DEVICES", "TP_SIZE", "EP_SIZE", "MOE_A2A_BACKEND", "PORT",
            "MEM_FRACTION_STATIC", "MAX_TOTAL_TOKENS", "CHUNKED_PREFILL_SIZE",
            "CUDA_GRAPH_BS_DECODE", "DISABLE_OVERLAP_SCHEDULE")},
        "arms": ["A1=0", "B1=2", "B2=2", "A2=0"],
        "warning": "A arms retain known Top-K nondeterminism; not a correctness baseline",
    }, indent=2))
    for arm, mode in (("A1", "0"), ("B1", "2"), ("B2", "2"), ("A2", "0")):
        env["SGLANG_DSV4_GFX90A_CANONICAL_INDEXER_ORDER"] = mode
        with (args.output_dir / f"{arm}_resources.txt").open("w") as out:
            subprocess.run(["amd-smi", "process", "--general", "--sort-by-pid"], stdout=out, stderr=subprocess.STDOUT, check=True)
        log = args.output_dir / f"{arm}_server.log"
        print(f"{arm}: launching independent TP4 service, Top-K mode={mode}", flush=True)
        with log.open("w") as out:
            process = subprocess.Popen(["bash", "scripts/rocm_dsv4_flash.sh", "serve"], cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 300
            while "The server is fired up and ready to roll!" not in log.read_text(errors="replace"):
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"{arm}: startup failed; see {log}")
                time.sleep(2)
            base = f"http://127.0.0.1:{args.port}"
            def run(name, script, options):
                print(f"{arm}: {name}", flush=True)
                with (args.output_dir / f"{arm}_{name}.stdout").open("w") as out:
                    subprocess.run([sys.executable, script, "--base-url", base,
                                    "--output", str(args.output_dir / f"{arm}_{name}.json"), *options],
                                   cwd=root, stdout=out, stderr=subprocess.STDOUT, timeout=600, check=True)
            c1_options = ["--arm", f"topk-{arm}-20260907", "--rounds", "3"]
            if arm != "A1":
                c1_options += ["--reference", str(args.output_dir / "A1_c1.json")]
            run("c1", "scripts/rocm/bench_dsv4_c1_mhc_recovery.py", c1_options)
            run("prefill_c1", "scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py",
                ["--request-count", "1", "--request-offset", "2", "--tokens", "1", "--rounds", "5"])
            run("prefill_c16", "scripts/rocm/bench_dsv4_prefill_diverse_concurrent.py",
                ["--request-count", "16", "--tokens", "1", "--rounds", "3"])
            print(f"{arm}: complete", flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=45)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
            # Allow this service's worker cleanup before checking the next arm.
            time.sleep(3)
    print("ABBA complete; all experiment services stopped", flush=True)


if __name__ == "__main__":
    main()
