#!/usr/bin/env python3
"""Real-code long-context factual smoke with explicit token/quality coverage.

Prepare the prompt once, then replay exactly those IDs through /generate.
This checks a few facts from real source/config, not general model quality or
equivalence to an independent full-model implementation.
"""

import argparse
import hashlib
import json
import math
import time
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCES = [
    "experimental/deepseek_v41/engram_host.py",
    "experimental/deepseek_v41/metadata.py",
    "python/sglang/srt/layers/attention/dsv4/dsv41_sparse.py",
    "python/sglang/kernels/ops/moe/gfx90a_dsv41_compact_ck.py",
]


def parse_answer(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    return json.loads(text)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--prepared", type=Path)
    p.add_argument("--source-file", type=Path, action="append")
    p.add_argument("--model-dir", type=Path, default=Path("/media/PM983/deepseek-v4.1-flash"))
    p.add_argument("--url", default="http://127.0.0.1:30101")
    p.add_argument("--min-input-tokens", type=int, default=16385)
    p.add_argument("--max-input-tokens", type=int, default=32000)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--timeout", type=float, default=1200)
    args = p.parse_args()
    if args.prepare_only == bool(args.prepared):
        p.error("choose --prepare-only or --prepared <artifact>")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    report = {"passed": False, "completed": False, "url": args.url}
    with args.output.open("x") as handle:
        try:
            if args.prepare_only:
                cfg = json.loads((args.model_dir / "config.json").read_text())["text_config"]
                fields = ["candidate_source_layer_id", "candidate_topk_blocks",
                          "candidate_block_size", "index_topk", "index_source_layer_ids",
                          "kv_source_layer_ids"]
                expected = {"candidate_source_layer": cfg["candidate_source_layer_id"],
                            "candidate_block_size": cfg["candidate_block_size"],
                            "candidate_budget_blocks": cfg["candidate_topk_blocks"],
                            "indexer_topk_positions": cfg["index_topk"],
                            "score_slab_mib": 128, "engram_tables_resident_on_gpu": False,
                            "tie_break": "lower_logical_id"}
                question = (
                    "Review the checkpoint configuration and real Python implementation below. "
                    "Return only a JSON object with exactly these fields: candidate_source_layer, "
                    "candidate_block_size, candidate_budget_blocks, indexer_topk_positions, "
                    "score_slab_mib, engram_tables_resident_on_gpu, tie_break. Use numbers for the "
                    "first five and a boolean for engram_tables_resident_on_gpu, referring to the "
                    "large tables in the RAM-backed Engram helper, not small gathered rows. "
                    "For tie_break choose exactly lower_logical_id, higher_logical_id, or unspecified "
                    "according to the stable indexer implementation. Derive answers from the provided "
                    "configuration and source. Do not summarize unrelated code or rewrite it.\n\n"
                    "Checkpoint configuration:\n```json\n" + json.dumps({k: cfg[k] for k in fields}, indent=2) + "\n```"
                )
                sources = args.source_file or [ROOT / x for x in DEFAULT_SOURCES]
                source_info = []
                for path in sources:
                    source = path.read_text()
                    source_info.append({"path": str(path), "sha256": hashlib.sha256(source.encode()).hexdigest()})
                    question += f"\n\nFile: {path.name}\n```python\n{source}\n```"
                question += "\n\nNow answer the seven requested fields as one JSON object."
                prompt = "<｜begin▁of▁sentence｜><｜User｜>" + question + "<｜Assistant｜></think>"
                ids = tokenizer.encode(prompt, add_special_tokens=False)
                report.update(input_ids=ids, input_tokens=len(ids), prompt=prompt,
                              sources=source_info, expected=expected, prepared=True)
                report["within_requested_length_range"] = args.min_input_tokens <= len(ids) <= args.max_input_tokens
                report["passed"] = report["within_requested_length_range"]
            else:
                prepared = json.loads(args.prepared.read_text())
                ids = prepared["input_ids"]
                if not prepared.get("prepared") or not args.min_input_tokens <= len(ids) <= args.max_input_tokens:
                    raise ValueError("prepared input does not satisfy requested long-context coverage")
                payload = {"input_ids": ids, "sampling_params": {"temperature": 0,
                           "max_new_tokens": args.max_new_tokens, "ignore_eos": False},
                           "cache_salt": f"dsv41-long-code-{time.time_ns()}",
                           "return_logprob": True, "logprob_start_len": -1, "top_logprobs_num": 20}
                report.update(prepared_artifact=str(args.prepared), request=payload,
                              input_ids=ids, input_tokens=len(ids), expected=prepared["expected"])
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                request = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(),
                                                 headers={"Content-Type": "application/json"})
                start = time.monotonic()
                try:
                    with opener.open(request, timeout=args.timeout) as response:
                        result = json.load(response)
                finally:
                    report["elapsed_s"] = time.monotonic() - start
                report.update(response=result, completed=True)
                output = result.get("output_ids")
                meta = result.get("meta_info", {})
                if not output or len(output) != meta.get("completion_tokens") or meta.get("cached_tokens") != 0:
                    raise ValueError("missing completion IDs or unexpected prefix cache hit")
                if meta.get("prompt_tokens") != len(ids):
                    raise ValueError("server did not report the complete long input")
                token_lp = meta.get("output_token_logprobs", [])
                top_lp = meta.get("output_top_logprobs", [])
                if len(token_lp) != len(output) or len(top_lp) != len(output):
                    raise ValueError("logprob coverage does not match completion IDs")
                for token, lp, top in zip(output, token_lp, top_lp):
                    if lp[1] != token or not math.isfinite(lp[0]) or len(top) != 20:
                        raise ValueError("invalid selected-token/top-20 logprobs")
                    if len({entry[1] for entry in top}) != 20 or not all(math.isfinite(entry[0]) for entry in top):
                        raise ValueError("invalid top-20 membership or nonfinite logprobs")
                decoded = tokenizer.decode(output, skip_special_tokens=True)
                if decoded != result["text"]:
                    raise ValueError("decoded completion does not match response text")
                actual = parse_answer(decoded)
                expected = prepared["expected"]
                exact_answer = (
                    isinstance(actual, dict) and actual == expected
                    and all(type(actual[key]) is type(value) for key, value in expected.items())
                )
                report.update(answer=actual, answer_matches=exact_answer,
                              completion_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
                report["passed"] = report["answer_matches"] and meta["finish_reason"]["type"] == "stop"
        except Exception as exc:
            report["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            json.dump(report, handle, ensure_ascii=False, indent=2)
    print(json.dumps({k: v for k, v in report.items() if k not in
                     ("input_ids", "prompt", "request", "response")}, indent=2), flush=True)
    if "response" in report:
        print(report["response"].get("text"), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
