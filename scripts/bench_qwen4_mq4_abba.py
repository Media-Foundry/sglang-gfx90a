#!/usr/bin/env python3
"""Concurrent, deterministic OpenAI benchmark for MQ4G128 ABBA arms."""

import argparse
import hashlib
import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


PROMPTS = [
    "Explain why the sky appears blue in a precise technical paragraph.",
    "Describe how a binary search works and state its time complexity.",
    "Write a concise paragraph about the formation of ocean currents.",
    "Explain the difference between latency and throughput in computing.",
    "Describe photosynthesis in one detailed paragraph.",
    "Explain why seasons occur on Earth without using bullet points.",
    "Give a compact technical explanation of virtual memory.",
    "Describe the water cycle in a coherent paragraph.",
    "Explain what a checksum detects and what it cannot guarantee.",
    "Describe how a heat pump transfers thermal energy.",
    "Explain the purpose of a database transaction in one paragraph.",
    "Describe the role of gravity in forming stars.",
    "Explain the distinction between precision and accuracy.",
    "Describe how public-key cryptography enables key exchange.",
    "Explain why ocean tides occur in a short technical paragraph.",
    "Describe how a compiler transforms source code into an executable.",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30001")
    parser.add_argument("--arm", required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    control = requests.Session()
    control.trust_env = False
    model = control.get(f"{args.url}/v1/models", timeout=10).json()["data"][0]["id"]

    def run_group(batch_size: int, round_id: int):
        barrier = threading.Barrier(batch_size + 1)

        def one(index: int):
            session = requests.Session()
            session.trust_env = False
            payload = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": PROMPTS[index]
                        + f"\nBenchmark nonce: {round_id}-{index}.",
                    }
                ],
                "max_tokens": args.max_tokens,
                "temperature": 0,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            barrier.wait()
            started = time.perf_counter()
            response = session.post(
                f"{args.url}/v1/chat/completions", json=payload, timeout=600
            )
            elapsed = time.perf_counter() - started
            response.raise_for_status()
            body = response.json()
            text = body["choices"][0]["message"]["content"]
            return {
                "index": index,
                "elapsed_s": elapsed,
                "tokens": body["usage"]["completion_tokens"],
                "finish": body["choices"][0]["finish_reason"],
                "sha256": hashlib.sha256(text.encode()).hexdigest(),
            }

        with ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [pool.submit(one, i) for i in range(batch_size)]
            barrier.wait()
            group_start = time.perf_counter()
            rows = [future.result() for future in futures]
            group_elapsed = time.perf_counter() - group_start
        tokens = sum(row["tokens"] for row in rows)
        return {
            "elapsed_s": group_elapsed,
            "tokens": tokens,
            "tok_s": tokens / group_elapsed,
            "requests": rows,
        }

    results = {"arm": args.arm, "model": model, "batches": {}}
    for batch_size in (1, 4, 8, 16):
        run_group(batch_size, -1)  # JIT and scheduler warmup, excluded.
        rounds = [run_group(batch_size, r) for r in range(args.rounds)]
        rates = [r["tok_s"] for r in rounds]
        trimmed = sorted(rates)[1:-1] if len(rates) >= 5 else rates
        results["batches"][str(batch_size)] = {
            "rounds": rounds,
            "median_tok_s": statistics.median(rates),
            "trimmed_mean_tok_s": statistics.fmean(trimmed),
        }
        print(
            f"{args.arm} BS{batch_size}: median={statistics.median(rates):.2f} "
            f"trim={statistics.fmean(trimmed):.2f} tok/s",
            flush=True,
        )
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
