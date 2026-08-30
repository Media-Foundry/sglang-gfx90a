#!/usr/bin/env python3
"""Collect a warm DeepSeek-V4 expert-occupancy recorder window.

The target server must be started with ``--expert-distribution-recorder-mode
stat`` and a buffer of at least ``warmup + window + tail + 2``.  The client
uses different, fixed token-ID prompts, records one concurrent native-AR
generation, and asks SGLang to dump the result.  Analyze the dump with
``.agents/memory/analyze_tp8_bs32_expert_occupancy.py``.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
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
    parser.add_argument("--warmup-passes", type=int, default=32)
    parser.add_argument("--window-passes", type=int, default=128)
    parser.add_argument("--request-count", type=int, default=32)
    parser.add_argument(
        "--tail-tokens",
        type=int,
        default=8,
        help="extra decode steps so the analyzer still has a full window",
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--dump-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--result-json", type=Path)
    return parser.parse_args()


def request_json(url: str, payload: dict | None, timeout: float) -> dict | str:
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=timeout) as response:
        body = response.read().decode()
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return body


def dump_files(directory: Path) -> set[Path]:
    return set(directory.glob("expert_distribution_recorder_*.pt"))


def main() -> None:
    args = parse_args()
    manifest = json.loads(args.inputs.read_text())
    requests = manifest["requests"]
    if len(requests) < args.request_count:
        raise ValueError(
            f"the occupancy corpus has {len(requests)} requests, "
            f"need {args.request_count}"
        )
    requests = requests[: args.request_count]
    ids = [tuple(item["input_ids"]) for item in requests]
    if len(set(ids)) != args.request_count:
        raise ValueError("all selected input_id sequences must be distinct")
    if any(not sequence or sequence[0] != 0 for sequence in ids):
        raise ValueError("every DSV4 request must start with BOS token 0")

    total_tokens = args.warmup_passes + args.window_passes + args.tail_tokens
    base_url = args.base_url.rstrip("/")
    before = dump_files(args.dump_dir)
    request_json(base_url + "/start_expert_distribution_record", None, args.timeout)

    barrier = threading.Barrier(args.request_count + 1)
    salt = time.time_ns()

    def generate(index: int, item: dict) -> tuple[float, dict]:
        payload = {
            "input_ids": item["input_ids"],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": total_tokens,
                "ignore_eos": True,
            },
            "cache_salt": f"occupancy-diverse-{index}-{salt}",
        }
        barrier.wait()
        begin = time.perf_counter()
        response = request_json(base_url + "/generate", payload, args.timeout)
        if not isinstance(response, dict):
            raise RuntimeError(f"request {index} returned a non-JSON response: {response!r}")
        return time.perf_counter() - begin, response

    error: BaseException | None = None
    results: list[tuple[float, dict]] = []
    begin = time.perf_counter()
    try:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.request_count
        ) as pool:
            futures = [
                pool.submit(generate, index, item)
                for index, item in enumerate(requests)
            ]
            barrier.wait()
            results = [future.result() for future in futures]
    except BaseException as exc:
        error = exc
    finally:
        request_json(base_url + "/stop_expert_distribution_record", None, args.timeout)
        request_json(base_url + "/dump_expert_distribution_record", None, args.timeout)
    wall = time.perf_counter() - begin
    if error is not None:
        raise error

    output_ids = [result.get("output_ids") or [] for _, result in results]
    lengths = [len(tokens) for tokens in output_ids]
    if lengths != [total_tokens] * args.request_count:
        raise RuntimeError(f"completion lengths are not all {total_tokens}: {lengths}")
    france_exact = output_ids[0][: len(FRANCE_EXPECTED)] == FRANCE_EXPECTED
    if not france_exact:
        raise RuntimeError(
            "France correctness oracle failed: "
            f"got={output_ids[0][:len(FRANCE_EXPECTED)]} expected={FRANCE_EXPECTED}"
        )

    deadline = time.monotonic() + 10
    new_dumps: list[Path] = []
    while time.monotonic() < deadline:
        new_dumps = sorted(
            dump_files(args.dump_dir) - before,
            key=lambda path: path.stat().st_mtime_ns,
        )
        if new_dumps:
            break
        time.sleep(0.1)
    if not new_dumps:
        raise RuntimeError(f"no new recorder dump appeared in {args.dump_dir}")

    result = {
        "format": "dsv4-diverse-occupancy-collection-v2",
        "base_url": base_url,
        "inputs": str(args.inputs.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "warmup_passes": args.warmup_passes,
        "window_passes": args.window_passes,
        "tail_tokens": args.tail_tokens,
        "request_count": args.request_count,
        "generated_tokens_per_request": total_tokens,
        "group_wall_seconds": wall,
        "aggregate_tokens_per_second": args.request_count * total_tokens / wall,
        "france_first9_exact": france_exact,
        "completion_sha256": [
            hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()
            for tokens in output_ids
        ],
        "request_wall_seconds": [request_wall for request_wall, _ in results],
        "recorder_dumps": [str(path.resolve()) for path in new_dumps],
    }
    encoded = json.dumps(result, indent=2) + "\n"
    if args.result_json:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        args.result_json.write_text(encoded)
    print(encoded, end="")
    print(
        "analyze with:\n  "
        f"python .agents/memory/analyze_tp8_bs32_expert_occupancy.py {new_dumps[-1]} "
        f"--warmup-passes {args.warmup_passes} --window-passes {args.window_passes} "
        "--csv /tmp/tp8_diverse_occupancy.csv --json /tmp/tp8_diverse_occupancy.json"
    )


if __name__ == "__main__":
    main()
