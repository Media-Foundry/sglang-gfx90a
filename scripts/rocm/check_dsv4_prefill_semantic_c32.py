#!/usr/bin/env python3
"""Generate and save semantic witnesses for the real heterogeneous C32 prefill."""

import concurrent.futures
import json
import os
import threading
import time
import urllib.request
from pathlib import Path


def request_one(url, item, index, barrier, nonce):
    payload = {
        "input_ids": item["input_ids"],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 64,
            "ignore_eos": False,
            "stream_interval": 1,
        },
        "cache_salt": f"prefill-semantic-{index}-{nonce}",
        "stream": True,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    barrier.wait()
    final = None
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(
        req, timeout=1200
    ) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            final = json.loads(body)
    if final is None:
        raise RuntimeError(f"request {index} returned no SSE payload")
    ids = final.get("output_ids") or []
    unique_ratio = len(set(ids)) / max(len(ids), 1)
    return {
        "index": index,
        "task": item.get("task"),
        "source": item.get("source"),
        "text": final.get("text"),
        "output_ids": ids,
        "completion_tokens": final.get("meta_info", {}).get("completion_tokens"),
        "finish_reason": final.get("meta_info", {}).get("finish_reason"),
        "unique_token_ratio": unique_ratio,
    }


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / ".agents/memory/dsv4_prefill_diverse_32_input_ids.json").read_text()
    )
    requests = manifest["requests"][:32]
    barrier = threading.Barrier(33)
    nonce = time.time_ns()
    url = os.getenv("SGLANG_BENCH_URL", "http://127.0.0.1:30001/generate")
    with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
        futures = [
            pool.submit(request_one, url, item, index, barrier, nonce)
            for index, item in enumerate(requests)
        ]
        barrier.wait()
        results = [future.result() for future in futures]
    output = Path("/tmp/dsv4_bf16_ck_c32_semantic.json")
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    for item in results:
        text = (item["text"] or "").replace("\n", " ")
        print(
            f'{item["index"]:02d} tokens={len(item["output_ids"]):2d} '
            f'unique={item["unique_token_ratio"]:.3f} task={item["task"]!r} '
            f'text={text[:180]!r}'
        )
    print(output)


if __name__ == "__main__":
    main()
