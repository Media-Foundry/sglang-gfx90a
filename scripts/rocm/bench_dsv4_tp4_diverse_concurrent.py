#!/usr/bin/env python3
"""Benchmark one TP4 service with 32 distinct, fixed DSV4 token-ID prompts."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


FRANCE_EXPECTED = [671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]
FRANCE_PREFIX = FRANCE_EXPECTED[:4]
FRANCE_PARIS_TOKEN = 11111


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--request-count",
        type=int,
        default=32,
        help="number of concurrent requests; 64 expands the 32-prompt manifest with unique continuations",
    )
    parser.add_argument(
        "--request-seed",
        type=int,
        help=(
            "deterministically sample and order the heterogeneous request set once; "
            "the France correctness sentinel remains request 0 and every round/arm "
            "reuses the resulting input_ids"
        ),
    )
    parser.add_argument(
        "--workload-output",
        type=Path,
        help="write the selected request set as a reusable input manifest",
    )
    parser.add_argument(
        "--stream-interval-sequence",
        type=str,
        help="comma-separated per-round intervals, e.g. 1,8,32,32,8,1",
    )
    parser.add_argument(
        "--stream-interval",
        type=int,
        default=1,
        help="tokens per streamed update; larger values reduce HTTP/host overhead",
    )
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument(
        "--allow-france-mismatch",
        action="store_true",
        help="record a semantically incorrect speculative baseline instead of aborting",
    )
    parser.add_argument(
        "--require-france-exact",
        action="store_true",
        help="require the historical nine-token oracle instead of the semantic Paris gate",
    )
    parser.add_argument(
        "--allow-no-resident-window",
        action="store_true",
        help="retain aggregate timing when staggered requests have no common window",
    )
    parser.add_argument(
        "--incremental-streaming-output",
        action="store_true",
        help="treat streamed output_ids as disjoint deltas",
    )
    parser.add_argument(
        "--position-bin-size",
        type=int,
        default=0,
        help="also report common resident decode windows in fixed token-position bins",
    )
    parser.add_argument(
        "--resident-time-bins",
        type=int,
        default=0,
        help="split the common BS32 resident wall-time window into equal bins",
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def get_url(url: str, timeout: float, as_json: bool = True, optional: bool = False):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=timeout) as response:
            body = response.read().decode()
    except urllib.error.HTTPError:
        if not optional:
            raise
        return None if as_json else ""
    return json.loads(body) if as_json else body


def decode_moments(snapshot: dict) -> list[float] | None:
    loads = snapshot.get("loads") or []
    if len(loads) != 1:
        return None
    value = loads[0].get("decode_moments")
    return [float(x) for x in value] if value else None


def metric_value(text: str, name: str, category: str = "decode") -> float | None:
    total = 0.0
    found = False
    for line in text.splitlines():
        if not line.startswith(name + "{") or f'category="{category}"' not in line:
            continue
        total += float(line.rsplit(None, 1)[1])
        found = True
    return total if found else None


def post_stream(
    url: str, payload: dict, timeout: float, incremental: bool = False
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
    accumulated_ids: list[int] = []
    with opener.open(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            body = line[len("data:") :].strip()
            if body == "[DONE]":
                break
            result = json.loads(body)
            if incremental:
                accumulated_ids.extend(result.get("output_ids") or [])
                result["output_ids"] = accumulated_ids.copy()
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


def select_requests(
    manifest_requests: list[dict], request_count: int, request_seed: int | None
) -> list[dict]:
    if request_seed is None:
        return list(manifest_requests[:request_count])

    france = next(
        (
            item
            for item in manifest_requests
            if item.get("prompt", "").strip() == "What is the capital of France?"
        ),
        None,
    )
    candidates = [item for item in manifest_requests if item is not france]
    keep = request_count - int(france is not None)
    if keep < 0 or keep > len(candidates):
        raise ValueError(
            f"cannot select {request_count} requests while pinning the France sentinel"
        )
    selected = random.Random(request_seed).sample(candidates, keep)
    return ([france] if france is not None else []) + selected


def main() -> None:
    args = parse_args()
    interval_sequence = None
    if args.stream_interval_sequence:
        interval_sequence = [
            int(value) for value in args.stream_interval_sequence.split(",")
        ]
        if not interval_sequence or any(value < 1 for value in interval_sequence):
            raise ValueError("stream intervals must be positive")
        args.rounds = len(interval_sequence)
    manifest_requests = json.loads(args.inputs.read_text())["requests"]
    if not 1 <= args.request_count <= min(64, len(manifest_requests)):
        raise ValueError(
            "--request-count must be between 1 and the manifest size "
            f"({len(manifest_requests)})"
        )
    if len(manifest_requests) < args.request_count:
        raise ValueError(
            f"input manifest has {len(manifest_requests)} requests, "
            f"need {args.request_count}"
        )
    requests = select_requests(
        manifest_requests, args.request_count, args.request_seed
    )
    if len({tuple(x["input_ids"]) for x in requests}) != len(requests):
        raise ValueError("selected input token sequences must all be distinct")
    first_is_france_oracle = requests[0].get("prompt", "").strip() == (
        "What is the capital of France?"
    )
    selected_requests_encoded = json.dumps(
        requests, sort_keys=True, separators=(",", ":")
    ).encode()
    selected_workload_sha256 = hashlib.sha256(selected_requests_encoded).hexdigest()
    if args.workload_output:
        args.workload_output.parent.mkdir(parents=True, exist_ok=True)
        args.workload_output.write_text(
            json.dumps(
                {
                    "format": "dsv4-diverse-input-ids-v1",
                    "request_seed": args.request_seed,
                    "requests": requests,
                },
                indent=2,
            )
            + "\n"
        )

    rounds = []
    round_output_ids: list[list[list[int]]] = []
    for rep in range(args.rounds):
        stream_interval = (
            interval_sequence[rep] if interval_sequence else args.stream_interval
        )
        loads_before = get_url(
            args.base_url.rstrip("/") + "/v1/loads?include=core", args.timeout
        )
        metrics_before = get_url(
            args.base_url.rstrip("/") + "/metrics",
            args.timeout,
            as_json=False,
            optional=True,
        )
        barrier = threading.Barrier(len(requests) + 1)
        nonce = time.time_ns()

        def generate(index: int, item: dict) -> tuple[float, dict, list[tuple[float, int]]]:
            payload = {
                "input_ids": item["input_ids"],
                "sampling_params": {
                    "temperature": args.temperature,
                    "max_new_tokens": args.tokens,
                    "ignore_eos": True,
                    "stream_interval": stream_interval,
                },
                "cache_salt": f"tp4-diverse-{rep}-{index}-{nonce}",
                "stream": True,
            }
            barrier.wait()
            begin = time.perf_counter()
            result, samples = post_stream(
                args.base_url.rstrip("/") + "/generate",
                payload,
                args.timeout,
                args.incremental_streaming_output,
            )
            return time.perf_counter() - begin, result, samples

        begin = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as pool:
            futures = [
                pool.submit(generate, index, item)
                for index, item in enumerate(requests)
            ]
            barrier.wait()
            results = [future.result() for future in futures]
        wall = time.perf_counter() - begin
        loads_after = get_url(
            args.base_url.rstrip("/") + "/v1/loads?include=core", args.timeout
        )
        metrics_after = get_url(
            args.base_url.rstrip("/") + "/metrics",
            args.timeout,
            as_json=False,
            optional=True,
        )
        ids = [completion_ids(result) for _, result, _ in results]
        round_output_ids.append(ids)
        lengths = [len(value) for value in ids]
        if lengths != [args.tokens] * len(requests):
            raise RuntimeError(f"round {rep}: completion lengths={lengths}")
        france_exact = (
            ids[0][: len(FRANCE_EXPECTED)] == FRANCE_EXPECTED
            if first_is_france_oracle
            else None
        )
        france_semantic = (
            ids[0][: len(FRANCE_PREFIX)] == FRANCE_PREFIX
            and FRANCE_PARIS_TOKEN in ids[0][:16]
            if first_is_france_oracle
            else None
        )
        france_gate = (
            france_exact if args.require_france_exact else france_semantic
        )
        if first_is_france_oracle and not france_gate and not args.allow_france_mismatch:
            raise RuntimeError(
                f"round {rep}: France oracle={ids[0][:len(FRANCE_EXPECTED)]} "
                f"semantic={france_semantic} exact={france_exact}"
            )
        finish_reasons = [
            result.get("meta_info", {}).get("finish_reason", {}).get("type")
            for _, result, _ in results
        ]
        spec_accept_lengths = [
            result.get("meta_info", {}).get("spec_accept_length")
            for _, result, _ in results
        ]
        spec_accept_rates = [
            result.get("meta_info", {}).get("spec_accept_rate")
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
        if (steady_wall <= 0 or steady_tokens <= 0) and not args.allow_no_resident_window:
            raise RuntimeError(
                f"round {rep}: no common resident decode window "
                f"wall={steady_wall} tokens={steady_tokens}"
            )
        has_resident_window = steady_wall > 0 and steady_tokens > 0
        position_bins = []
        if args.position_bin_size > 0:
            for lo in range(0, args.tokens, args.position_bin_size):
                hi = min(args.tokens, lo + args.position_bin_size)

                def time_at_count(samples, count):
                    for sample_time, sample_count in samples:
                        if sample_count >= count:
                            return sample_time
                    raise RuntimeError(
                        f"round {rep}: no timestamp for completion count {count}"
                    )

                bin_start = max(time_at_count(samples, lo) for samples in token_samples)
                bin_end = min(time_at_count(samples, hi) for samples in token_samples)
                bin_tokens = sum(
                    count_at(samples, bin_end) - count_at(samples, bin_start)
                    for samples in token_samples
                )
                if bin_end > bin_start and bin_tokens > 0:
                    position_bins.append(
                        {
                            "start": lo,
                            "end": hi,
                            "wall_s": bin_end - bin_start,
                            "tokens": bin_tokens,
                            "tok_s": bin_tokens / (bin_end - bin_start),
                        }
                    )
        resident_time_bins = []
        if args.resident_time_bins > 0:
            for bin_id in range(args.resident_time_bins):
                bin_start = steady_start + steady_wall * bin_id / args.resident_time_bins
                bin_end = steady_start + steady_wall * (bin_id + 1) / args.resident_time_bins
                bin_tokens = sum(
                    count_at(samples, bin_end) - count_at(samples, bin_start)
                    for samples in token_samples
                )
                bin_events = sum(
                    sum(bin_start < sample_time <= bin_end for sample_time, _ in samples)
                    for samples in token_samples
                )
                resident_time_bins.append(
                    {
                        "bin": bin_id,
                        "wall_s": bin_end - bin_start,
                        "tokens": bin_tokens,
                        "tok_s": bin_tokens / (bin_end - bin_start),
                        "stream_events": bin_events,
                        "events_s": bin_events / (bin_end - bin_start),
                        "tokens_per_event": (
                            bin_tokens / bin_events if bin_events > 0 else None
                        ),
                    }
                )
        moments_before = decode_moments(loads_before)
        moments_after = decode_moments(loads_after)
        moments_delta = (
            [b - a for a, b in zip(moments_before, moments_after)]
            if moments_before and moments_after
            else None
        )
        gpu_seconds_before = metric_value(
            metrics_before, "sglang:forward_execution_seconds_total"
        )
        gpu_seconds_after = metric_value(
            metrics_after, "sglang:forward_execution_seconds_total"
        )
        gpu_seconds_delta = (
            gpu_seconds_after - gpu_seconds_before
            if gpu_seconds_before is not None and gpu_seconds_after is not None
            else None
        )
        record = {
            "round": rep,
            "request_count": len(requests),
            "stream_interval": stream_interval,
            "group_wall_s": wall,
            "aggregate_tok_s": sum(lengths) / wall,
            "resident_bs32_wall_s": steady_wall if has_resident_window else None,
            "resident_bs32_tokens": steady_tokens if has_resident_window else None,
            "resident_bs32_tok_s": (
                steady_tokens / steady_wall if has_resident_window else None
            ),
            "position_bins": position_bins,
            "resident_time_bins": resident_time_bins,
            "lengths": lengths,
            "finish_reasons": finish_reasons,
            "spec_accept_length_mean": (
                statistics.mean(
                    value for value in spec_accept_lengths if value is not None
                )
                if any(value is not None for value in spec_accept_lengths)
                else None
            ),
            "spec_accept_rate_mean": (
                statistics.mean(value for value in spec_accept_rates if value is not None)
                if any(value is not None for value in spec_accept_rates)
                else None
            ),
            "spec_accept_lengths": spec_accept_lengths,
            "france_first9_exact": france_exact,
            "france_semantic_paris": france_semantic,
            "completion_sha256": [
                hashlib.sha256(
                    json.dumps(value, separators=(",", ":")).encode()
                ).hexdigest()
                for value in ids
            ],
            # Compact first-divergence witness without storing every generated
            # token from every BS32 round.
            "completion_first16_ids": [value[:16] for value in ids],
            "request_wall_s": [elapsed for elapsed, _, _ in results],
            "decode_moments_delta": moments_delta,
            "decode_step_count": moments_delta[0] if moments_delta else None,
            "scheduler_decode_tok_s": (
                moments_delta[5] / (moments_delta[2] / 1e6)
                if moments_delta and moments_delta[2] > 0
                else None
            ),
            "host_mean_decode_step_ms": (
                moments_delta[2] / moments_delta[0] / 1e3
                if moments_delta and moments_delta[0] > 0
                else None
            ),
            "gpu_forward_seconds_delta": gpu_seconds_delta,
            "gpu_mean_forward_ms": (
                gpu_seconds_delta * 1e3 / moments_delta[0]
                if gpu_seconds_delta is not None
                and moments_delta
                and moments_delta[0] > 0
                else None
            ),
        }
        rounds.append(record)
        print(json.dumps(record, separators=(",", ":")), flush=True)

    speeds = [item["aggregate_tok_s"] for item in rounds]
    trimmed = sorted(speeds)[1:-1] if len(speeds) > 2 else speeds
    resident_speeds = [
        item["resident_bs32_tok_s"]
        for item in rounds
        if item["resident_bs32_tok_s"] is not None
    ]
    resident_trimmed = (
        sorted(resident_speeds)[1:-1]
        if len(resident_speeds) > 2
        else resident_speeds
    )

    def first_divergence(sequences: list[list[int]]) -> int | None:
        reference = sequences[0]
        for position in range(len(reference)):
            if any(
                sequence[position] != reference[position]
                for sequence in sequences[1:]
            ):
                return position
        return None

    per_request_sequences = [
        [round_ids[request_index] for round_ids in round_output_ids]
        for request_index in range(len(requests))
    ]
    first_divergence_by_request = [
        first_divergence(sequences) for sequences in per_request_sequences
    ]
    cross_round_exact_requests = sum(
        divergence is None for divergence in first_divergence_by_request
    )
    summary = {
        "format": "dsv4-tp4-diverse-concurrent-v1",
        "input_manifest": str(args.inputs.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "request_seed": args.request_seed,
        "selected_workload_sha256": selected_workload_sha256,
        "workload_output": (
            str(args.workload_output.resolve()) if args.workload_output else None
        ),
        "first_request_is_france_oracle": first_is_france_oracle,
        "tokens": args.tokens,
        "temperature": args.temperature,
        "require_france_exact": args.require_france_exact,
        "request_count": len(requests),
        "stream_interval": args.stream_interval,
        "stream_interval_sequence": interval_sequence,
        "incremental_streaming_output": args.incremental_streaming_output,
        "round_count": len(rounds),
        "median_tok_s": statistics.median(speeds),
        "trimmed_mean_tok_s": statistics.mean(trimmed),
        "resident_bs32_median_tok_s": (
            statistics.median(resident_speeds) if resident_speeds else None
        ),
        "resident_bs32_trimmed_mean_tok_s": (
            statistics.mean(resident_trimmed) if resident_trimmed else None
        ),
        "cross_round_exact_requests": cross_round_exact_requests,
        "cross_round_all_exact": cross_round_exact_requests == len(requests),
        "first_divergence_by_request": first_divergence_by_request,
        "rounds": rounds,
    }
    encoded = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="", flush=True)


if __name__ == "__main__":
    main()
