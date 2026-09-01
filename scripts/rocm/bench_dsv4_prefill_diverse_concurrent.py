#!/usr/bin/env python3
"""Measure C1--C32 DSV4 prefill on fixed, real heterogeneous code prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=root / ".agents/memory/dsv4_prefill_diverse_32_input_ids.json",
    )
    parser.add_argument("--request-count", type=int, choices=(1, 4, 8, 16, 32), default=32)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def post_stream(url: str, payload: dict, timeout: float) -> tuple[dict, float, float, float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    begin = time.perf_counter()
    first = None
    result = {}
    with opener.open(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            result = json.loads(body)
            if first is None:
                first = time.perf_counter()
    end = time.perf_counter()
    if first is None:
        raise RuntimeError("stream returned no token")
    return result, begin, first, end


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.inputs.read_text())
    selected = manifest["requests"][: args.request_count]
    if len(selected) != args.request_count:
        raise ValueError("manifest has too few requests")
    if len({tuple(item["input_ids"]) for item in selected}) != len(selected):
        raise ValueError("prompts must be distinct")
    url = args.base_url.rstrip("/") + "/generate"
    rounds = []
    output_witnesses = []
    for rep in range(args.rounds):
        barrier = threading.Barrier(args.request_count + 1)
        nonce = time.time_ns()

        def run(index: int, item: dict):
            payload = {
                "input_ids": item["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.tokens,
                    "ignore_eos": True,
                    "stream_interval": 1,
                },
                "cache_salt": f"prefill-diverse-{rep}-{index}-{nonce}",
                "stream": True,
            }
            barrier.wait()
            return post_stream(url, payload, args.timeout)

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.request_count) as pool:
            futures = [pool.submit(run, i, item) for i, item in enumerate(selected)]
            barrier.wait()
            results = [future.result() for future in futures]
        first_begin = min(item[1] for item in results)
        last_first = max(item[2] for item in results)
        last_end = max(item[3] for item in results)
        prompt_tokens = [
            int(result[0].get("meta_info", {}).get("prompt_tokens", 0))
            for result in results
        ]
        completion_ids = [result[0].get("output_ids") or [] for result in results]
        if prompt_tokens != [item["prompt_tokens"] for item in selected]:
            raise RuntimeError(
                f"server prompt counts differ from manifest: {prompt_tokens}"
            )
        if [len(ids) for ids in completion_ids] != [args.tokens] * args.request_count:
            raise RuntimeError("one or more requests did not complete")
        prefill_wall = last_first - first_begin
        total_prompt = sum(prompt_tokens)
        record = {
            "round": rep,
            "request_count": args.request_count,
            "total_prompt_tokens": total_prompt,
            "prefill_wall_s": prefill_wall,
            "aggregate_input_tok_s": total_prompt / prefill_wall,
            "group_wall_s": last_end - first_begin,
            "completion_lengths": [len(ids) for ids in completion_ids],
            "completion_sha256": [
                hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                for ids in completion_ids
            ],
        }
        rounds.append(record)
        output_witnesses.append(completion_ids)
        print(json.dumps(record, separators=(",", ":")), flush=True)
    speeds = [item["aggregate_input_tok_s"] for item in rounds]
    trimmed = sorted(speeds)[1:-1] if len(speeds) > 2 else speeds
    summary = {
        "format": "dsv4-prefill-diverse-concurrent-v1",
        "input_manifest": str(args.inputs.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "request_count": args.request_count,
        "tokens": args.tokens,
        "rounds": rounds,
        "median_input_tok_s": statistics.median(speeds),
        "trimmed_mean_input_tok_s": statistics.mean(trimmed),
        "cross_round_first_token_exact": all(
            output_witnesses[rep][req][0] == output_witnesses[0][req][0]
            for rep in range(1, len(output_witnesses))
            for req in range(args.request_count)
        ),
    }
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
