#!/usr/bin/env python3
"""Repeat a real-source C1 response without forcing post-EOS continuation."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30011")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--tokens", type=int, default=1024)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    source = json.loads((root / ".agents/memory/dsv4_prefill_diverse_32_input_ids.json").read_text())["requests"][args.case_index]
    result = {"input_ids": source["input_ids"], "task": source["task"],
              "responses": [], "status": "running"}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for rep in range(2):
        payload = {
            "input_ids": source["input_ids"],
            "sampling_params": {"temperature": 0, "max_new_tokens": args.tokens},
            "cache_salt": f"woa-source-{uuid.uuid4().hex}-{rep}",
        }
        request = urllib.request.Request(
            args.base_url + "/generate", data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        with opener.open(request, timeout=180) as response:
            out = json.load(response)
        elapsed = time.perf_counter() - start
        count = out["meta_info"]["completion_tokens"]
        assert count == len(out["output_ids"]) and count > 0
        assert out["meta_info"]["cached_tokens"] == 0
        digest = hashlib.sha256(json.dumps(out["output_ids"]).encode()).hexdigest()
        result["responses"].append({"response": out, "wall_s": elapsed, "sha256": digest})
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
        print(rep, count, elapsed, digest, flush=True)
    result["status"] = "complete"
    result["repeat_ids_exact"] = (result["responses"][0]["response"]["output_ids"]
                                  == result["responses"][1]["response"]["output_ids"])
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n")
    print("repeat_ids_exact", result["repeat_ids_exact"], flush=True)


if __name__ == "__main__":
    main()
