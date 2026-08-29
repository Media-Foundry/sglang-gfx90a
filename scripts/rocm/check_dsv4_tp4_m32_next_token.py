#!/usr/bin/env python3
"""Capture one teacher-forced M32 next-token/logprob oracle from /generate."""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30001")
    parser.add_argument(
        "--inputs",
        type=Path,
        default=root / ".agents/memory/dsv4_tp8_diverse_32_input_ids.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    requests = json.loads(args.inputs.read_text())["requests"]
    if len(requests) != 32 or len({tuple(x["input_ids"]) for x in requests}) != 32:
        raise ValueError("oracle requires exactly 32 distinct input-ID prompts")
    payload = {
        "input_ids": [item["input_ids"] for item in requests],
        "sampling_params": {
            "temperature": 0,
            "max_new_tokens": 1,
            "ignore_eos": True,
        },
        "return_logprob": True,
        "logprob_start_len": -1,
        "top_logprobs_num": 5,
        "stream": False,
    }
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request, timeout=1200) as response:
        results = json.loads(response.read())
    if not isinstance(results, list) or len(results) != 32:
        raise RuntimeError(f"expected 32 results, got {type(results)}")

    rows = []
    for result in results:
        meta = result.get("meta_info", {})
        rows.append(
            {
                "output_ids": result.get("output_ids"),
                "output_token_logprobs": meta.get("output_token_logprobs"),
                "output_top_logprobs": meta.get("output_top_logprobs"),
            }
        )
    record = {
        "format": "dsv4-tp4-m32-next-token-v1",
        "input_manifest": str(args.inputs.resolve()),
        "input_manifest_sha256": hashlib.sha256(args.inputs.read_bytes()).hexdigest(),
        "rows": rows,
    }
    encoded = json.dumps(record, indent=2) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "output_ids": [row["output_ids"] for row in rows],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
