#!/usr/bin/env python3
"""E2E drift trial for the non-bitwise fused MHC paths on strict C32 DSpark.

Both candidate paths are faster and neither is bitwise equal to the split path,
so a hash comparison alone cannot judge them: the strict C32 control is itself
recorded as `cross_round_all_exact=false`. This harness therefore measures
candidate divergence against a control-vs-control noise floor, in ABBA arm
order (A1 B1 B2 A2) so drift and throughput are read from the same runs.

It does not start or stop services -- the caller points it at one already-warm
service per arm, so an arm's env is fixed at service launch.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MANIFEST = Path("/tmp/dsv4_open_code_pd_20260908/decode.json")


def load_manifest(path: Path, count: int) -> list[dict]:
    payload = json.loads(path.read_text())
    if payload.get("phase") != "decode":
        raise ValueError(f"{path} is not a decode manifest")
    requests = payload["requests"][:count]
    if len(requests) < count:
        raise ValueError(f"manifest has {len(requests)} requests, need {count}")
    return requests


def completion_hash(token_ids: list[int]) -> str:
    joined = ",".join(str(t) for t in token_ids)
    return hashlib.sha256(joined.encode()).hexdigest()[:16]


def post(url: str, body: dict, timeout: float) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with opener.open(request, timeout=timeout) as response:
        return json.loads(response.read())


def get(url: str, timeout: float = 30.0) -> dict:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=timeout) as response:
        return json.loads(response.read())


def accept_lengths(base_url: str) -> list[float]:
    """Per-DP `avg_spec_accept_length` from /server_info's live scheduler state."""
    try:
        states = get(base_url + "/server_info").get("internal_states") or []
    except Exception:
        return []
    return [
        state["avg_spec_accept_length"]
        for state in states
        if state.get("avg_spec_accept_length") is not None
    ]


def run_wave(base_url: str, requests: list[dict], tokens: int, wave: str,
             stable_salt: bool, timeout: float) -> dict:
    """One synchronized C32 wave. Greedy, fixed length, ignore_eos."""
    barrier = threading.Barrier(len(requests) + 1)
    # A fixed salt namespace per wave keeps page placement comparable between
    # arms; a nonce would confound cache effects with numerical drift.
    nonce = 0 if stable_salt else time.time_ns()
    results: dict[int, dict] = {}

    def one(item: dict) -> None:
        body = {
            "input_ids": item["input_ids"],
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": tokens,
                "ignore_eos": True,
            },
            "cache_salt": f"mhc-trial-{wave}-{item['index']}-{nonce}",
            "return_logprob": False,
        }
        barrier.wait()
        out = post(base_url + "/generate", body, timeout)
        meta = out.get("meta_info", {})
        # A non-stream finished response carries output_ids at the top level.
        # Refuse an empty list: `compare` would read it as perfect agreement
        # and report a false no-drift result for the whole trial.
        ids = meta.get("output_ids") or out.get("output_ids") or []
        if not ids:
            raise RuntimeError(
                f"request {item['index']} returned no output_ids; "
                f"keys={sorted(out)} meta_keys={sorted(meta)}"
            )
        results[item["index"]] = {
            "output_ids": ids,
            "text": out.get("text", ""),
            "completion_tokens": meta.get("completion_tokens"),
            "hash": completion_hash(ids),
        }

    with ThreadPoolExecutor(max_workers=len(requests)) as pool:
        futures = [pool.submit(one, item) for item in requests]
        barrier.wait()
        begin = time.perf_counter()
        for future in futures:
            future.result()
        wall = time.perf_counter() - begin

    produced = sum(len(r["output_ids"]) for r in results.values())
    return {
        "wall_s": wall,
        "produced_tokens": produced,
        "aggregate_tok_s": produced / wall if wall else 0.0,
        # A candidate must not buy step time by accepting fewer draft tokens.
        "accept_lengths": accept_lengths(base_url),
        "per_request": {str(k): v for k, v in sorted(results.items())},
    }


def compare(left: dict, right: dict) -> dict:
    """Per-request divergence between two waves.

    `first_diff` is the earliest differing token position, so a late cosmetic
    divergence is distinguishable from an immediate semantic one.
    """
    indices = sorted(set(left["per_request"]) & set(right["per_request"]))
    differing, first_diffs, prefix_ratios = [], [], []
    for index in indices:
        a = left["per_request"][index]["output_ids"]
        b = right["per_request"][index]["output_ids"]
        if a == b:
            prefix_ratios.append(1.0)
            continue
        differing.append(index)
        limit = min(len(a), len(b))
        position = next((i for i in range(limit) if a[i] != b[i]), limit)
        first_diffs.append(position)
        prefix_ratios.append(position / max(len(a), len(b), 1))
    return {
        "requests": len(indices),
        "differing": len(differing),
        "differing_indices": differing,
        "exact_fraction": (len(indices) - len(differing)) / max(len(indices), 1),
        "first_diff_min": min(first_diffs) if first_diffs else None,
        "first_diff_median": statistics.median(first_diffs) if first_diffs else None,
        "mean_shared_prefix_fraction": statistics.fmean(prefix_ratios) if prefix_ratios else 1.0,
    }


def repetition_flag(text: str, window: int = 48, repeats: int = 4) -> bool:
    """Severe-repetition gate: same window recurring back-to-back."""
    if len(text) < window * repeats:
        return False
    for size in (window, window // 2, window // 4):
        for start in range(0, len(text) - size * repeats + 1, size):
            chunk = text[start:start + size]
            if chunk and chunk * repeats == text[start:start + size * repeats]:
                return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True,
                        help="already-warm service for this arm")
    parser.add_argument("--arm", required=True, choices=["A1", "B1", "B2", "A2"],
                        help="ABBA position; A=control, B=candidate")
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--waves", type=int, default=2,
                        help="measured waves per arm; >=2 gives an in-arm floor")
    parser.add_argument("--warm-tokens", type=int, default=128)
    parser.add_argument("--stable-salt", action="store_true", default=True)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    items = load_manifest(args.manifest, args.requests)
    # /server_info spreads the resolved server args at the top level rather
    # than nesting them, and /get_server_info is a deprecated alias for it.
    info = get(args.base_url + "/server_info")
    observed = {
        "tp_size": info.get("tp_size"),
        "speculative_algorithm": info.get("speculative_algorithm"),
        "speculative_num_steps": info.get("speculative_num_steps"),
        "max_total_tokens": info.get("max_total_tokens"),
        "cuda_graph_max_bs": info.get("cuda_graph_max_bs"),
    }
    if observed["tp_size"] != 8:
        raise SystemExit(f"expected TP8, got tp_size={observed['tp_size']}")
    if not (observed["speculative_algorithm"] or "").lower().count("dspark"):
        raise SystemExit(
            f"expected a DSpark service, got {observed['speculative_algorithm']!r}"
        )
    print(f"ARM {args.arm} service={observed}", flush=True)

    # Excluded warm wave: first-touch JIT and page placement must not land in a
    # measured wave.
    run_wave(args.base_url, items, args.warm_tokens, f"{args.arm}-warm",
             args.stable_salt, args.timeout)

    waves = []
    for index in range(args.waves):
        wave = run_wave(args.base_url, items, args.tokens,
                        f"{args.arm}-w{index}", args.stable_salt, args.timeout)
        repeats = [k for k, v in wave["per_request"].items()
                   if repetition_flag(v["text"])]
        wave["severe_repetition"] = repeats
        waves.append(wave)
        print(f"  wave{index}: {wave['aggregate_tok_s']:.2f} tok/s "
              f"({wave['produced_tokens']} tok in {wave['wall_s']:.1f}s) "
              f"repetition={len(repeats)}", flush=True)

    in_arm = [compare(waves[i], waves[i + 1]) for i in range(len(waves) - 1)]
    for index, cmp in enumerate(in_arm):
        print(f"  in-arm w{index}->w{index+1}: differing "
              f"{cmp['differing']}/{cmp['requests']}, "
              f"first_diff_median={cmp['first_diff_median']}", flush=True)

    payload = {
        "arm": args.arm,
        "base_url": args.base_url,
        "server": observed,
        "manifest": str(args.manifest),
        "requests": args.requests,
        "tokens": args.tokens,
        "waves": waves,
        "in_arm_comparisons": in_arm,
        "aggregate_tok_s": [w["aggregate_tok_s"] for w in waves],
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(f"WROTE {args.output}", flush=True)


if __name__ == "__main__":
    main()
