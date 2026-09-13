#!/usr/bin/env python3
"""Compare cached decode with fresh-cache recomputation of identical prefixes.

Uses real baseline generated tokens as teacher-forced continuation for each
recompute. It never compares two independently sampled trajectories. Reports
top-logprob diagnostics rather than claiming bitwise floating-point parity.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path


def validate(out, topn):
    ids = out.get("output_ids")
    meta = out.get("meta_info", {})
    if not ids or len(ids) != meta.get("completion_tokens"):
        raise ValueError("missing/empty or non-completion-only output IDs")
    scores = meta.get("output_top_logprobs")
    token_lp = meta.get("output_token_logprobs")
    if not isinstance(scores, list) or len(scores) != len(ids):
        raise ValueError("top-logprob coverage does not match completion IDs")
    if not isinstance(token_lp, list) or len(token_lp) != len(ids):
        raise ValueError("selected-token logprobs are missing")
    for token, lp, row in zip(ids, token_lp, scores):
        if lp[1] != token or not math.isfinite(lp[0]):
            raise ValueError("selected-token logprob alignment mismatch")
        if len(row) != topn or len({v[1] for v in row}) != topn:
            raise ValueError("invalid top-logprob row")
        if not all(math.isfinite(v[0]) for v in row):
            raise ValueError("non-finite logprob")
    return ids, scores


def compare(a, b):
    aa, bb = {v[1]: v[0] for v in a}, {v[1]: v[0] for v in b}
    common = aa.keys() & bb.keys()
    return {
        "cached_top1": a[0][1], "recompute_top1": b[0][1],
        "top1_equal": a[0][1] == b[0][1],
        "cached_top1_margin": a[0][0] - a[1][0],
        "recompute_top1_margin": b[0][0] - b[1][0],
        "top_overlap": len(common) / len(a),
        "max_abs_common_logprob_delta": max((abs(aa[t] - bb[t]) for t in common), default=None),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompt-artifact", type=Path, required=True, help="JSON containing the exact input_ids to use")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30101")
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--prefix-lengths", default="68,69,70,127,128,129,255,256,257")
    p.add_argument("--timeout", type=float, default=240)
    p.add_argument("--topn", type=int, default=20)
    p.add_argument("--baseline", type=Path, help="Reuse a recorded baseline response; recomputes are still fresh requests")
    p.add_argument("--repeats", type=int, default=1)
    args = p.parse_args()
    if args.repeats < 1 or args.topn < 2 or args.max_new_tokens < 1:
        p.error("--repeats/max-new-tokens must be positive and --topn at least 2")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    prompt = json.loads(args.prompt_artifact.read_text())["input_ids"]
    assert prompt and all(type(x) is int and x >= 0 for x in prompt)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def generate(prefix, max_tokens, name):
        payload = {
            "input_ids": prefix,
            "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens, "ignore_eos": False},
            "cache_salt": f"dsv41-prefix-{time.time_ns()}",
            "return_logprob": True, "logprob_start_len": -1,
            "top_logprobs_num": args.topn,
        }
        start = time.monotonic()
        record = {"request": payload, "completed": False}
        try:
            req = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(),
                                         headers={"Content-Type": "application/json"})
            with opener.open(req, timeout=args.timeout) as response:
                record["response"] = json.load(response)
            record["completed"] = True
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record["elapsed_s"] = time.monotonic() - start
            with (args.output_dir / f"{name}.json").open("x") as f:
                json.dump(record, f, indent=2, ensure_ascii=False)
        validate(record["response"], args.topn)
        if record["response"]["meta_info"].get("cached_tokens") != 0:
            raise ValueError("fresh-cache recompute unexpectedly reused a prefix")
        return record["response"]

    report = {"prompt_tokens": len(prompt), "comparisons": [], "coverage_complete": False}
    try:
        if args.baseline:
            base_record = json.loads(args.baseline.read_text())
            if base_record["request"]["input_ids"] != prompt:
                raise ValueError("reused baseline has a different prompt")
            baseline = base_record["response"]
            report["reused_baseline"] = str(args.baseline)
        else:
            baseline = generate(prompt, args.max_new_tokens, "cached-baseline")
        ids, baseline_scores = validate(baseline, args.topn)
        report["baseline_completion_tokens"] = len(ids)
        report["baseline_completion_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        print(f"baseline finished: {len(ids)} completion tokens", flush=True)
        for length in [int(s) for s in args.prefix_lengths.split(",")]:
            step = length - len(prompt)
            if not 0 <= step < len(ids):
                raise ValueError(f"baseline does not cover prefix length {length}")
            for repeat in range(args.repeats):
                recompute = generate(prompt + ids[:step], 1, f"recompute-{length}-{repeat}")
                next_ids, next_scores = validate(recompute, args.topn)
                row = {"prefix_length": length, "generated_offset": step, "repeat": repeat,
                       "cached_committed_id": ids[step], "recompute_committed_id": next_ids[0],
                       **compare(baseline_scores[step], next_scores[0])}
                report["comparisons"].append(row)
                print(json.dumps(row), flush=True)
        report["coverage_complete"] = True
        report["all_top1_equal"] = all(x["top1_equal"] for x in report["comparisons"])
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        with (args.output_dir / "summary.json").open("x") as f:
            json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["coverage_complete"] and report["all_top1_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
