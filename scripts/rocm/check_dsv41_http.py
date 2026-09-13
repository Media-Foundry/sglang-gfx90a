#!/usr/bin/env python3
"""Proxy-free HTTP/semantic smoke for V4.1 bring-up (not a quality benchmark).

Uses explicit checkpoint-tokenizer IDs, records the unmodified responses, and
fails rather than interpreting missing output IDs as a successful empty hash.
The France sentinel proves only that sentinel, not general model correctness.
"""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30101")
    parser.add_argument("--model-dir", default="/media/PM983/deepseek-v4.1-flash")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--user-message", default="What is the capital of France? Answer in one short sentence.")
    parser.add_argument("--expect", default="Paris", help="Case-insensitive substring smoke, not a quality oracle")
    parser.add_argument("--source-file", type=Path, action="append", default=[], help="Append real source text to the user message")
    args = parser.parse_args()
    # Refuse to overwrite an earlier trial, including a partially failed one.
    with args.output.open("x") as record:
        report = {"url": args.url, "model_dir": args.model_dir, "passed": False}
        try:
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

            def request(path, payload=None):
                req = urllib.request.Request(
                    args.url.rstrip("/") + path,
                    data=None if payload is None else json.dumps(payload).encode(),
                    headers={"Content-Type": "application/json"},
                )
                with opener.open(req, timeout=args.timeout) as response:
                    return json.load(response)

            report["models"] = request("/v1/models")
            if not report["models"].get("data"):
                raise ValueError("/v1/models returned no model entries")
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                args.model_dir, trust_remote_code=True, local_files_only=True
            )
            user_message = args.user_message
            for source in args.source_file:
                user_message += f"\n\nFile: {source.name}\n```\n{source.read_text()}\n```"
            prompt = (
                "<｜begin▁of▁sentence｜><｜User｜>" + user_message
                + "<｜Assistant｜></think>"
            )
            input_ids = tokenizer.encode(prompt, add_special_tokens=False)
            report.update(prompt=prompt, input_ids=input_ids, bos_count=input_ids.count(0))
            payload = {
                "input_ids": input_ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                    "ignore_eos": False,
                },
                "cache_salt": f"dsv41-smoke-{time.time_ns()}",
            }
            start = time.monotonic()
            out = request("/generate", payload)
            report.update(elapsed_s=time.monotonic() - start, response=out)
            ids = out.get("output_ids")
            count = out.get("meta_info", {}).get("completion_tokens")
            if not isinstance(ids, list) or not ids or not isinstance(count, int):
                raise ValueError("Missing/empty output_ids or completion_tokens")
            if count <= 0 or len(ids) != count:
                raise ValueError(f"output_ids length={len(ids)} != completion_tokens={count}")
            if not all(isinstance(token, int) and 0 <= token < len(tokenizer) for token in ids):
                raise ValueError("Invalid output token IDs")
            report["completion_sha256"] = hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest()
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            report["decoded_completion"] = decoded
            report["text_matches_ids"] = decoded == out.get("text")
            report["expected_substring"] = args.expect
            report["expected_substring_pass"] = args.expect.lower() in decoded.lower()
            report["passed"] = report["text_matches_ids"] and report["expected_substring_pass"]
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            json.dump(report, record, ensure_ascii=False, indent=2)
            record.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
