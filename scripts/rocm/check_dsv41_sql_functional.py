#!/usr/bin/env python3
"""Varied long-output coding checks against read-only, in-memory SQL fixtures.

Never executes Python/shell from model output. SQLite authorizer rejects
writes, external databases, pragmas and unapproved functions; a progress
budget interrupts runaway recursive queries. Explanations are retained but
only the SQL result is functionally graded. This is not a standard benchmark
score or an independent full-model numerical reference.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from check_dsv41_cache_prefix import validate
from dsv41_sql_cases import CASES, fixture, prompt_for


PURE_FUNCTIONS = {
    "abs", "avg", "coalesce", "count", "dense_rank", "first_value", "ifnull",
    "iif", "lag", "last_value", "lead", "length", "lower", "max", "min",
    "nullif", "rank", "round", "row_number", "sum", "total", "upper",
}


def extract_sql(text):
    blocks = re.findall(r"```sql[ \t]*\r?\n(.*?)```", text, flags=re.I | re.S)
    if len(blocks) != 1:
        raise ValueError("expected exactly one fenced SQL query")
    return blocks[0].strip()


def check_query(name, query, seeds=16, vm_callbacks=1000):
    if seeds < 2 or vm_callbacks < 1:
        raise ValueError("empty/tiny test sets cannot certify a query")
    task = CASES[name]
    reports = []
    for seed in range(seeds):
        rows, expected = fixture(name, seed)
        record = {"seed": seed, "input_rows": rows, "expected": expected, "passed": False}
        connection = sqlite3.connect(":memory:")
        try:
            connection.enable_load_extension(False)
            connection.execute(task["schema"])
            if rows:
                slots = ",".join("?" for _ in rows[0])
                connection.executemany(f"INSERT INTO {task['table']} VALUES ({slots})", rows)
            connection.commit()

            def authorize(action, arg1, arg2, _database, _origin):
                if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_READ and arg1 == task["table"]:
                    return sqlite3.SQLITE_OK
                if action == sqlite3.SQLITE_FUNCTION and (arg2 or "").lower() in PURE_FUNCTIONS:
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY

            callbacks = 0
            deadline = time.monotonic() + 1.0

            def progress():
                nonlocal callbacks
                callbacks += 1
                return int(callbacks >= vm_callbacks or time.monotonic() > deadline)

            connection.set_authorizer(authorize)
            connection.set_progress_handler(progress, 1000)
            actual = connection.execute(query).fetchmany(1001)
            record.update(actual=actual, passed=actual == expected, progress_callbacks=callbacks)
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            connection.close()
        reports.append(record)
    return {"passed": all(row["passed"] for row in reports), "fixtures": reports}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--url", default="http://127.0.0.1:30101")
    p.add_argument("--model-dir", default="/media/PM983/deepseek-v4.1-flash")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--concurrency", type=int, choices=[1, 4], default=4)
    p.add_argument("--max-new-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=900)
    args = p.parse_args()
    if args.rounds < 1 or args.max_new_tokens < 1 or args.timeout <= 0:
        p.error("rounds, max-new-tokens and timeout must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, local_files_only=True, trust_remote_code=True)
    prepared = {}
    for name in CASES:
        prompt = "<｜begin▁of▁sentence｜><｜User｜>" + prompt_for(name) + "<｜Assistant｜></think>"
        prepared[name] = {"prompt": prompt, "input_ids": tokenizer.encode(prompt, add_special_tokens=False)}
    with (args.output_dir / "prepared.json").open("x") as f:
        json.dump(prepared, f, indent=2, ensure_ascii=False)
    results = []
    for round_id in range(args.rounds):
        barrier = threading.Barrier(args.concurrency)

        def request(name):
            payload = {"input_ids": prepared[name]["input_ids"],
                       "sampling_params": {"temperature": 0, "max_new_tokens": args.max_new_tokens,
                                           "ignore_eos": False},
                       "return_logprob": True, "top_logprobs_num": 20, "logprob_start_len": -1,
                       "cache_salt": f"dsv41-sql-{round_id}-{name}-{time.time_ns()}"}
            row = {"case": name, "round": round_id, "request": payload,
                   "passed": False, "completed": False}
            barrier.wait(timeout=30)
            start = time.monotonic()
            try:
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                req = urllib.request.Request(args.url + "/generate", data=json.dumps(payload).encode(),
                                             headers={"Content-Type": "application/json"})
                with opener.open(req, timeout=args.timeout) as response:
                    out = json.load(response)
                row.update(completed=True, response=out, http_elapsed_s=time.monotonic() - start)
                ids, _ = validate(out, 20)
                meta = out["meta_info"]
                if meta["prompt_tokens"] != len(payload["input_ids"]) or meta.get("cached_tokens") != 0:
                    raise ValueError("input accounting or fresh-cache contract violated")
                decoded = tokenizer.decode(ids, skip_special_tokens=True)
                if decoded != out["text"] or meta["finish_reason"]["type"] != "stop":
                    raise ValueError("text/ID mismatch or output did not reach natural EOS")
                query = extract_sql(decoded)
                evaluation = check_query(name, query)
                row.update(query=query, evaluation=evaluation, passed=evaluation["passed"],
                           completion_tokens=len(ids),
                           completion_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest())
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            with (args.output_dir / f"round{round_id}-{name}.json").open("x") as f:
                json.dump(row, f, ensure_ascii=False, indent=2)
            print(json.dumps({k: row.get(k) for k in ["case", "round", "completed", "passed",
                                                     "completion_tokens", "http_elapsed_s", "error"]}), flush=True)
            return row

        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            results.extend(pool.map(request, CASES))
    summary = {"passed": len(results) == len(CASES) * args.rounds and all(r["passed"] for r in results),
               "requests": len(results), "concurrency": args.concurrency, "rounds": args.rounds,
               "semantic_passes": sum(r["passed"] for r in results), "cases": {}}
    for name in CASES:
        group = [r for r in results if r["case"] == name]
        summary["cases"][name] = {"token_lengths": [r.get("completion_tokens") for r in group],
                                  "hashes": [r.get("completion_sha256") for r in group],
                                  "passed": [r["passed"] for r in group]}
    with (args.output_dir / "summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
