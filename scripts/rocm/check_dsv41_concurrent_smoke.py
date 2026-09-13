#!/usr/bin/env python3
"""Small, varied C4 correctness probes, not a throughput benchmark."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import urllib.request

CASES = [
    ("france", "What is the capital of France? Answer in one short sentence.", "paris"),
    ("arithmetic", "Compute 19 times 23. Reply with the integer only.", "437"),
    ("python", "In Python, how do I reverse the list xs using slicing? Reply with the expression only.", "xs[::-1]"),
    ("sql", "Write a SQL query that counts all rows in the events table. Reply with the query only.", "count(*)"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30101")
    p.add_argument("--rounds", type=int, default=2)
    args = p.parse_args()
    if args.rounds < 1:
        p.error("--rounds must be positive; an empty run is not a pass")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("/media/PM983/deepseek-v4.1-flash", local_files_only=True, trust_remote_code=True)
    prepared = [(name, tokenizer.encode("<｜begin▁of▁sentence｜><｜User｜>" + prompt + "<｜Assistant｜></think>", add_special_tokens=False), expect) for name, prompt, expect in CASES]
    all_results = []
    for r in range(args.rounds):
        barrier = threading.Barrier(len(prepared))

        def request(case):
            name, ids, expect = case
            payload = {"input_ids": ids, "sampling_params": {"temperature": 0, "max_new_tokens": 48, "ignore_eos": False}, "cache_salt": f"c4-smoke-{r}-{name}-{time.time_ns()}"}
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            barrier.wait(timeout=30)
            start = time.monotonic()
            result = {"case": name, "round": r, "request": payload, "passed": False}
            try:
                req = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                with opener.open(req, timeout=120) as response:
                    out = json.load(response)
                result["response"] = out
                output_ids = out.get("output_ids")
                if not output_ids or len(output_ids) != out.get("meta_info", {}).get("completion_tokens"):
                    raise ValueError("missing/non-completion-only output IDs")
                decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
                result["text_matches_ids"] = decoded == out["text"]
                result["expected_substring"] = expect
                result["expected_substring_pass"] = expect in decoded.lower().replace(" ", "")
                result["hash"] = hashlib.sha256(json.dumps(output_ids).encode()).hexdigest()
                result["passed"] = result["text_matches_ids"] and result["expected_substring_pass"] and out["meta_info"]["finish_reason"]["type"] == "stop"
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {exc}"
            result["elapsed_s"] = time.monotonic() - start
            with (args.output_dir / f"round{r}-{name}.json").open("x") as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(json.dumps({k: result.get(k) for k in ("case", "round", "passed", "elapsed_s", "hash", "error")}), flush=True)
            return result

        with ThreadPoolExecutor(max_workers=4) as executor:
            all_results.extend(executor.map(request, prepared))
    summary = {"passed": all(x["passed"] for x in all_results), "rounds": args.rounds, "requests": len(all_results),
               "hashes_per_case": {name: sorted({x.get("hash", "ERROR") for x in all_results if x["case"] == name}) for name, _, _ in CASES}}
    with (args.output_dir / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
