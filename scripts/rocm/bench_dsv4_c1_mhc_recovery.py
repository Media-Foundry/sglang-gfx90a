#!/usr/bin/env python3
"""Fixed C1 code workload and teacher-forced probes for MHC recovery ABBA.

Use the same inputs, arm ordering and warmup for every fresh server. No GPU
allocation or tokenizer required in the client. Outputs are diagnostic data,
not a proof of executable code correctness or full model equivalence.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct
import threading
import time
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-url", default="http://127.0.0.1:30011")
    p.add_argument("--arm", required=True)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", type=Path)
    p.add_argument("--skip-freeze-gc", action="store_true")
    p.add_argument("--smoke-only", action="store_true",
                   help="2304-token real-source prefill plus four concurrent short requests")
    args = p.parse_args()
    nonce = uuid.uuid4().hex
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / ".agents/memory/dsv4_tp8_diverse_32_input_ids.json").read_text()
    )["requests"]
    cases = [r for r in manifest if r["id"] in ("diverse-03", "diverse-15", "diverse-31")]
    op = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(payload, endpoint="/generate"):
        req = urllib.request.Request(
            args.base_url + endpoint,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        t = time.perf_counter()
        with op.open(req, timeout=180) as response:
            data = response.read()
        return data, time.perf_counter() - t

    if not args.skip_freeze_gc:
        post({}, "/freeze_gc")
    result = {"arm": args.arm, "measurements": [], "teacher_forced": [],
              "status": "running"}

    def checkpoint():
        # Outside request timing. Preserve completed evidence if a later
        # independent probe times out; never label an incomplete run complete.
        temporary = args.output.with_suffix(args.output.suffix + ".partial")
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
        temporary.replace(args.output)

    def generate(case, rep, tokens):
        raw, elapsed = post({
            "input_ids": case["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": tokens, "ignore_eos": True},
            "cache_salt": f"mhc-recovery-{nonce}-{args.arm}-{rep}-{case['id']}",
        })
        out = json.loads(raw)
        n = out["meta_info"]["completion_tokens"]
        ids = out["output_ids"]
        assert n == tokens and len(ids) == n, (n, tokens, len(ids))
        assert out["meta_info"]["cached_tokens"] == 0, out["meta_info"]
        return {
            "case": case["id"], "prompt": case["prompt"], "rep": rep,
            "tokens": n, "wall_s": elapsed, "tok_s": n / elapsed,
            "output_ids": ids, "text": out["text"],
            "sha256": hashlib.sha256(struct.pack(f"<{n}I", *ids)).hexdigest(),
            "finish": out["meta_info"]["finish_reason"],
            "cached_tokens": out["meta_info"]["cached_tokens"],
        }

    france = manifest[0]
    for rep in range(2):
        row = generate(france, f"france-{rep}", 9)
        assert row["output_ids"] == [671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1], row
    result["france_exact"] = True
    checkpoint()
    if args.smoke_only:
        source = json.loads(
            (root / ".agents/memory/dsv4_prefill_diverse_32_input_ids.json").read_text()
        )["requests"][0]
        long_case = {"id": "source-2304", "prompt": source["task"],
                     "input_ids": source["input_ids"]}
        for rep in range(3):
            row = generate(long_case, f"prefill-{rep}", 1)
            result["measurements"].append(row)
            print(args.arm, "prefill", rep, row["wall_s"], row["output_ids"], flush=True)
        for rep in range(2):
            barrier = threading.Barrier(4)

            def concurrent(case):
                barrier.wait()
                # Stop the sentinel at its EOS; forcing it to 64 tokens tests
                # undefined post-EOS continuation rather than answer quality.
                tokens = 9 if case["id"] == france["id"] else 64
                return generate(case, f"concurrent-{rep}", tokens)

            with ThreadPoolExecutor(max_workers=4) as pool:
                rows = list(pool.map(concurrent, [france] + cases))
            assert rows[0]["output_ids"][:9] == [671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]
            result["measurements"].extend(rows)
            print(args.arm, "C4", rep, [r["sha256"][:16] for r in rows], flush=True)
        result["status"] = "complete"
        checkpoint()
        return
    for rep in range(-1, args.rounds):
        for case in cases:
            row = generate(case, rep, 256)
            result["measurements"].append(row)
            checkpoint()
            print(args.arm, rep, case["id"], round(row["tok_s"], 3), row["sha256"][:16], flush=True)

    # Feed identical continuation IDs to every arm: never compare logprobs
    # from independently generated prefixes. The A1 artifact owns the corpus.
    reference = json.loads(args.reference.read_text()) if args.reference else result
    for case in cases:
        source = next(r for r in reference["measurements"] if r["case"] == case["id"] and r["rep"] == 0)
        for length in (32, 128):
            ids = case["input_ids"] + source["output_ids"][:length]
            raw, elapsed = post({
                "input_ids": ids,
                "sampling_params": {"temperature": 0, "max_new_tokens": 1},
                "cache_salt": f"mhc-forced-{nonce}-{args.arm}-{case['id']}-{length}",
                "return_logprob": True, "logprob_start_len": len(case["input_ids"]),
                "top_logprobs_num": 20,
            })
            out = json.loads(raw)
            meta = out["meta_info"]
            for value, *_ in meta["input_token_logprobs"]:
                assert value is None or math.isfinite(value), meta
            result["teacher_forced"].append({
                "case": case["id"], "continuation_length": length, "input_ids": ids,
                "input_token_logprobs": meta["input_token_logprobs"],
                "output_top_logprobs": meta["output_top_logprobs"],
                "output_ids": out["output_ids"], "wall_s": elapsed,
            })
            checkpoint()
    result["medians"] = {
        case["id"]: statistics.median(r["tok_s"] for r in result["measurements"]
                                       if r["case"] == case["id"] and r["rep"] >= 0)
        for case in cases
    }
    result["status"] = "complete"
    checkpoint()
    print(json.dumps(result["medians"]), flush=True)


if __name__ == "__main__":
    main()
