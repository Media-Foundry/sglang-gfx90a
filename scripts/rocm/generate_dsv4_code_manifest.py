#!/usr/bin/env python3
"""Generate a reproducible 32-request DSV4 coding-workload manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from sglang.srt.entrypoints.openai import encoding_dsv4


PROMPTS = [
    "Implement a Python async retry helper with exponential backoff, jitter, cancellation, and unit tests.",
    "Write a Python function that reverses a singly linked list in place and include two unit tests.",
    "Implement an LRU cache in Rust without unsafe code and explain the ownership choices.",
    "Debug this JavaScript symptom: an async loop returns before all fetch calls finish. Show a minimal fix.",
    "Design a PostgreSQL query that returns each customer's latest paid invoice without a correlated subquery.",
    "Write a Bash script that finds files larger than 1 GiB, handles spaces safely, and sorts them by size.",
    "Implement binary search in C++20 using iterators and state the loop invariant.",
    "Write a Go worker pool with context cancellation, bounded concurrency, and clean shutdown.",
    "Refactor a Python web handler so database transactions roll back on every exception path.",
    "Implement a lock-free single-producer single-consumer ring buffer in C11 and explain memory ordering.",
    "Write a HIP kernel that performs a wave64 reduction of BF16 values into FP32 and discuss alignment.",
    "Optimize a matrix transpose kernel for CDNA2 using LDS tiling while avoiding bank conflicts.",
    "Given a CUDA kernel with uncoalesced column reads, rewrite its indexing for coalesced memory access.",
    "Implement stable top-k selection for 512 candidates and return original indices on equal scores.",
    "Write a Triton kernel for row-wise RMSNorm and identify which dimensions should be compile-time constants.",
    "Implement Dijkstra's algorithm in TypeScript with a binary heap and return the shortest path itself.",
    "Create a property-based test for a JSON parser that targets Unicode escapes and deeply nested arrays.",
    "Diagnose a distributed all-reduce that intermittently hangs only during graph capture; propose instrumentation.",
    "Write a CMakeLists.txt that builds a HIP executable and links a locally installed Composable Kernel library.",
    "Implement a streaming CSV parser in Java that does not load the entire input into memory.",
    "Design a Redis rate limiter that remains correct under concurrent clients and clock skew.",
    "Write a SQL migration that adds a non-null column to a large production table without a long blocking lock.",
    "Implement a recursive-descent parser for arithmetic expressions in Kotlin, including unary minus.",
    "Explain and fix an ABA bug in a compare-and-swap based free list with a concrete code sketch.",
    "Write a NumPy reference implementation of grouped-query attention with a causal mask.",
    "Implement a deterministic parallel prefix sum in HIP and explain the inter-block synchronization strategy.",
    "Review a REST API that retries POST requests and propose an idempotency-key design with failure cases.",
    "Write a fuzz harness for a C function that decodes variable-length integers and guard every overflow.",
    "Implement a persistent segment tree in modern C++ and demonstrate two historical range-sum queries.",
    "Design an asyncio pipeline with backpressure between download, parse, and database-write stages.",
    "Write a minimal eBPF program that counts TCP connects per process and describe the userspace reader.",
    "Analyze a GPU inference trace containing hundreds of 2-microsecond kernels and propose a safe fusion order.",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/pc/models/modelscope"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".agents/memory/dsv4_tp4_code_32_input_ids.json"),
    )
    args = parser.parse_args()

    tokenizer_path = args.model / "tokenizer.json"
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    requests = []
    for index, prompt in enumerate(PROMPTS):
        encoded = encoding_dsv4.encode_messages(
            [
                {"role": "system", "content": ""},
                {"role": "user", "content": prompt},
            ],
            thinking_mode="thinking",
        )
        input_ids = tokenizer.encode(encoded)
        requests.append(
            {"id": f"code-{index:02d}", "prompt": prompt, "input_ids": input_ids}
        )

    payload = {
        "format": "dsv4-official-thinking-code-fixed-input-ids-v1",
        "tokenizer_json_sha256": hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {len(requests)} distinct coding requests to {args.output}")


if __name__ == "__main__":
    main()
