#!/usr/bin/env python3
"""Benchmark one TP4 service with 32 distinct, fixed DSV4 token-ID prompts."""

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


FRANCE_EXPECTED = [671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=root / ".agents/memory/dsv4_tp8_diverse_32_input_ids.json",
    )
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def post_stream(
    url: str, payload: dict, timeout: float
) -> tuple[dict, list[tuple[float, int]]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result: dict = {}
    samples: list[tuple[float, int]] = []
    last_count = -1
    with opener.open(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            result = json.loads(body)
            count = len(result.get("output_ids") or [])
            if count > last_count:
                samples.append((time.perf_counter(), count))
                last_count = count
    if not isinstance(result, dict):
        raise RuntimeError(f"non-object response: {result!r}")
    if not samples:
        raise RuntimeError("stream returned no output token samples")
    return result, samples


def completion_ids(result: dict) -> list[int]:
    ids = result.get("output_ids")
    if not isinstance(ids, list):
        raise RuntimeError(f"response has no output_ids list: {result.keys()}")
    completion_tokens = result.get("meta_info", {}).get("completion_tokens", len(ids))
    if len(ids) != completion_tokens:
        ids = ids[-completion_tokens:]
    return ids


def main() -> None:
    args = parse_args()
    requests = json.loads(args.inputs.read_text())["requests"]
    if len(requests) != 32 or len({tuple(x["input_ids"]) for x in requests}) != 32:
        raise ValueError("input manifest must contain exactly 32 distinct requests")

    rounds = []
    for rep in range(args.rounds):
        barrier = threading.Barrier(33)
        nonce = time.time_ns()

        def generate(index: int, item: dict) -> tuple[float, dict, list[tuple[float, int]]]:
            payload = {
                "input_ids": item["input_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.tokens,
                    "ignore_eos": True,
                },
                "cache_salt": f"tp4-diverse-{rep}-{index}-{nonce}",
                "stream": True,
            }
            barrier.wait()
            begin = time.perf_counter()
            result, samples = post_stream(
                args.base_url.rstrip("/") + "/generate", payload, args.timeout
            )
            return time.perf_counter() - begin, result, samples

        begin = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as pool:
            futures = [
                pool.submit(generate, index, item)
                for index, item in enumerate(requests)
            ]
            barrier.wait()
            results = [future.result() for future in futures]
        wall = time.perf_counter() - begin
        ids = [completion_ids(result) for _, result, _ in results]
        lengths = [len(value) for value in ids]
        if lengths != [args.tokens] * 32:
            raise RuntimeError(f"round {rep}: completion lengths={lengths}")
        if ids[0][: len(FRANCE_EXPECTED)] != FRANCE_EXPECTED:
            raise RuntimeError(
                f"round {rep}: France oracle={ids[0][:len(FRANCE_EXPECTED)]}"
            )
        finish_reasons = [
            result.get("meta_info", {}).get("finish_reason", {}).get("type")
            for _, result, _ in results
        ]
        token_samples = [samples for _, _, samples in results]
        steady_start = max(samples[0][0] for samples in token_samples)
        steady_end = min(samples[-1][0] for samples in token_samples)

        def count_at(samples: list[tuple[float, int]], timestamp: float) -> int:
            count = 0
            for sample_time, sample_count in samples:
                if sample_time > timestamp:
                    break
                count = sample_count
            return count

        steady_tokens = sum(
            count_at(samples, steady_end) - count_at(samples, steady_start)
            for samples in token_samples
        )
        steady_wall = steady_end - steady_start
        if steady_wall <= 0 or steady_tokens <= 0:
            raise RuntimeError(
                f"round {rep}: no common resident decode window "
                f"wall={steady_wall} tokens={steady_tokens}"
            )
        record = {
            "round": rep,
            "group_wall_s": wall,
            "aggregate_tok_s": sum(lengths) / wall,
            "resident_bs32_wall_s": steady_wall,
            "resident_bs32_tokens": steady_tokens,
            "resident_bs32_tok_s": steady_tokens / steady_wall,
            "lengths": lengths,
            "finish_reasons": finish_reasons,
            "france_first9_exact": True,
            "completion_sha256": [
                hashlib.sha256(
                    json.dumps(value, separators=(",", ":")).encode()
                ).hexdigest()
                for value in ids
            ],
            "request_wall_s": [elapsed for elapsed, _, _ in results],
        }
        rounds.append(record)
        print(json.dumps(record, separators=(",", ":")), flush=True)

    speeds = [item["aggregate_tok_s"] for item in rounds]
    trimmed = sorted(speeds)[1:-1] if len(speeds) > 2 else speeds
    resident_speeds = [item["resident_bs32_tok_s"] for item in rounds]
    resident_trimmed = (
        sorted(resident_speeds)[1:-1]
        if len(resident_speeds) > 2
        else resident_speeds
    )
    summary = {
        "format": "dsv4-tp4-diverse-concurrent-v1",
        "input_manifest": str(args.inputs.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "tokens": args.tokens,
        "round_count": len(rounds),
        "median_tok_s": statistics.median(speeds),
        "trimmed_mean_tok_s": statistics.mean(trimmed),
        "resident_bs32_median_tok_s": statistics.median(resident_speeds),
        "resident_bs32_trimmed_mean_tok_s": statistics.mean(resident_trimmed),
        "rounds": rounds,
    }
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
