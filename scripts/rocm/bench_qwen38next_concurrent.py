#!/usr/bin/env python3
"""Concurrent native-AR decode benchmark for the gfx90a Qwen3.8Next profile."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import http.client
import json
import statistics
import threading
import time
import urllib.parse


PROMPTS = [
    "Explain why the sky appears blue in two concise sentences.",
    "Give a short proof that the square root of two is irrational.",
    "Describe binary search and state its time complexity.",
    "What is the capital of France? Answer in one sentence.",
    "Explain the difference between latency and throughput.",
    "Write one paragraph about error-correcting codes.",
    "Describe the role of a cache in a computer system.",
    "Explain why sorting can improve locality of reference.",
    "Summarize Newton's first law in plain language.",
    "Give three practical uses of matrix multiplication.",
    "Explain what a hash function does without using equations.",
    "Describe the producer-consumer pattern in one paragraph.",
    "Why does batching often improve GPU utilization?",
    "Explain the difference between precision and accuracy.",
    "Describe a balanced binary tree in two sentences.",
    "Explain why deterministic tests help optimize systems.",
    "Give a concise definition of virtual memory.",
    "Describe how a compiler lowers source code to machine code.",
    "Explain why synchronization can limit parallel speedup.",
    "What is arithmetic intensity? Give a concise answer.",
    "Describe one advantage and one cost of quantization.",
    "Explain the purpose of an attention mechanism.",
    "Describe a race condition with a simple example.",
    "Explain the difference between bandwidth and latency.",
    "Give a concise explanation of dynamic programming.",
    "Describe why memory alignment matters on a GPU.",
    "Explain what a reduction operation is in parallel computing.",
    "Describe the purpose of a page table in one paragraph.",
    "Explain the idea behind mixture-of-experts models.",
    "Why can small matrix shapes underutilize a GPU?",
    "Describe the difference between prefill and decode inference.",
    "Explain why reproducible benchmarks need repeated trials.",
]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def completion_hash(output_ids: list[int]) -> str:
    payload = json.dumps(output_ids, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def first_divergence(actual: list[int], expected: list[int]) -> int | None:
    for index, (lhs, rhs) in enumerate(zip(actual, expected)):
        if lhs != rhs:
            return index
    if len(actual) != len(expected):
        return min(len(actual), len(expected))
    return None


def send(url: str, payload: dict, barrier: threading.Barrier, timeout: int):
    target = urllib.parse.urlsplit(url)
    body = json.dumps(payload).encode()
    barrier.wait()
    begin = time.perf_counter()
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=timeout)
    try:
        connection.request(
            "POST", target.path, body=body, headers={"Content-Type": "application/json"}
        )
        response = connection.getresponse()
        response_body = response.read()
        if response.status != 200:
            raise RuntimeError(
                f"HTTP {response.status}: {response_body[:512].decode(errors='replace')}"
            )
        result = json.loads(response_body)
    finally:
        connection.close()
    return time.perf_counter() - begin, result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30001/generate")
    parser.add_argument("--concurrency", type=int, choices=(16, 32), default=32)
    parser.add_argument("--tokens", type=int, default=256)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args()

    reference_hashes: list[str] | None = None
    reference_outputs: list[list[int]] | None = None
    measured_rates = []
    measured_hash_exact = []
    for rep in range(args.warmups + args.reps):
        barrier = threading.Barrier(args.concurrency + 1)
        nonce = time.time_ns()
        payloads = [
            {
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.tokens,
                    "ignore_eos": True,
                },
                "cache_salt": f"qwen-concurrent-{args.concurrency}-{rep}-{i}-{nonce}",
            }
            for i, prompt in enumerate(PROMPTS[: args.concurrency])
        ]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as pool:
            futures = [
                pool.submit(send, args.url, payload, barrier, max(1200, args.tokens * 4))
                for payload in payloads
            ]
            barrier.wait()
            group_begin = time.perf_counter()
            results = [future.result() for future in futures]
            group_wall = time.perf_counter() - group_begin

        walls = [wall for wall, _ in results]
        outputs = [body.get("output_ids") or [] for _, body in results]
        lengths = [len(ids) for ids in outputs]
        hashes = [completion_hash(ids) for ids in outputs]
        finishes = [body.get("meta_info", {}).get("finish_reason") for _, body in results]
        phase = "warmup" if rep < args.warmups else "measure"
        if phase == "measure" and reference_hashes is None:
            # JIT/admission warmup can use a different batch composition.  The
            # first measured steady round is the trajectory reference.
            reference_hashes = hashes
            reference_outputs = outputs
        hash_exact = None if reference_hashes is None else hashes == reference_hashes
        divergences = (
            None
            if reference_outputs is None
            else [
                first_divergence(actual, expected)
                for actual, expected in zip(outputs, reference_outputs)
            ]
        )
        length_exact = all(length == args.tokens for length in lengths)
        aggregate = sum(lengths) / group_wall
        if phase == "measure":
            measured_rates.append(aggregate)
            measured_hash_exact.append(bool(hash_exact))
        print(
            json.dumps(
                {
                    "phase": phase,
                    "rep": rep,
                    "concurrency": args.concurrency,
                    "tokens": args.tokens,
                    "group_wall_s": round(group_wall, 6),
                    "aggregate_tok_s": round(aggregate, 3),
                    "request_wall_p50_s": round(statistics.median(walls), 6),
                    "request_wall_p95_s": round(percentile(walls, 0.95), 6),
                    "length_exact": length_exact,
                    "hash_exact": hash_exact,
                    "first_divergence": divergences,
                    "lengths": lengths,
                    "hashes": hashes,
                    "finish": finishes,
                },
                separators=(",", ":"),
            ),
            flush=True,
        )

    if measured_rates:
        trimmed = sorted(measured_rates)
        if len(trimmed) >= 5:
            trimmed = trimmed[1:-1]
        print(
            json.dumps(
                {
                    "summary": True,
                    "concurrency": args.concurrency,
                    "median_tok_s": round(statistics.median(measured_rates), 3),
                    "trimmed_mean_tok_s": round(statistics.mean(trimmed), 3),
                    "rates": [round(rate, 3) for rate in measured_rates],
                    "hash_stable_rounds": sum(measured_hash_exact),
                    "hash_measured_rounds": len(measured_hash_exact),
                },
                separators=(",", ":"),
            )
        )


if __name__ == "__main__":
    main()
