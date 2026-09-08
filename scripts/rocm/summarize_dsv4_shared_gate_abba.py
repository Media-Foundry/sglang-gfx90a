#!/usr/bin/env python3
"""Validate scoped TP8 shared-gate/attention ABBA and summarize warm rates."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import struct


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    state = json.loads(args.state.read_text())
    assert state["status"] == "complete", "ABBA is not complete"
    assert [b["candidate"] for b in state["blocks"]] == [True, False, False, True]
    assert state["candidate_flag"] in (
        "SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND",
        "SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2",
        "SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2",
        "SGLANG_DSV4_GFX90A_TP8_M32_SHARED_AFTER_TOPK",
        "SGLANG_DSV4_GFX90A_TP8_M32_DEFERRED_FINALIZE",
    )
    blocks = []
    reference = None
    forced_reference = None
    c1_rounds = state.get('c1_rounds', 2)
    assert isinstance(c1_rounds, int) and c1_rounds >= 2
    for block in state["blocks"]:
        assert block["status"] == "complete"
        c1 = json.loads(Path(block["c1"]).read_text())
        c32 = json.loads(Path(block["c32"]).read_text())
        assert c1["status"] == "complete" and c1["france_exact"]
        measured = [r for r in c1["measurements"] if r["rep"] >= 0]
        assert len(measured) == 3*c1_rounds
        expected_cases = {'diverse-03', 'diverse-15', 'diverse-31'}
        assert {(r['case'], r['rep']) for r in measured} == {
            (case, rep) for case in expected_cases for rep in range(c1_rounds)}
        if reference is None:
            reference = {r["case"]: r["output_ids"] for r in measured}
            forced_reference = {(r["case"], r["continuation_length"]): r
                                for r in c1["teacher_forced"]}
        for row in measured:
            ids = row["output_ids"]
            assert len(ids) == 256 and ids == reference[row["case"]]
            assert hashlib.sha256(struct.pack("<256I", *ids)).hexdigest() == row["sha256"]
        forced_exact = {}
        for field in ("output_ids", "input_token_logprobs", "output_top_logprobs"):
            forced_exact[field] = sum(
                r[field] == forced_reference[(r["case"], r["continuation_length"])][field]
                for r in c1["teacher_forced"]
            )
        if state["candidate_flag"] in ("SGLANG_DSV4_GFX90A_TP8_DECODE_ATTN_WARPS2",
                                       "SGLANG_DSV4_GFX90A_TP8_C1_ATTN_WARPS2"):
            assert len(c1["teacher_forced"]) == 6
            assert all(n == 6 for n in forced_exact.values()), forced_exact
        assert c32["request_count"] == 32 and c32["tokens"] == 256
        assert len(c32["rounds"]) == 6
        for row in c32["rounds"]:
            assert row["lengths"] == [256] * 32
            assert row["finish_reasons"] == ["length"] * 32
            assert len(row["output_ids"]) == len(row["completion_sha256"]) == 32
            for ids, digest in zip(row["output_ids"], row["completion_sha256"]):
                assert len(ids) == 256
                assert hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() == digest
        warm = c32["rounds"][1:]
        cases = sorted(reference)
        medians = [statistics.median(r["tok_s"] for r in measured if r["case"] == case)
                   for case in cases]
        case_samples = {case: [r['tok_s'] for r in measured if r['case'] == case]
                        for case in cases}
        trimmed = [statistics.mean(sorted(values)[1:-1] if len(values) >= 4 else values)
                   for values in case_samples.values()]
        blocks.append({
            "index": block["index"], "candidate": block["candidate"],
            "service_pid": block["service_pid"],
            "c1_tok_s": statistics.geometric_mean(medians),
            "c1_trimmed_tok_s": statistics.geometric_mean(trimmed),
            "c1_case_samples": case_samples,
            "c32_warm_tok_s": statistics.median(r["aggregate_tok_s"] for r in warm),
            "resident_warm_tok_s": statistics.median(r["resident_bs32_tok_s"] for r in warm),
            "c1_ids_exact": len(measured), "teacher_forced_exact": forced_exact,
            "c32_cross_round_exact": c32["cross_round_exact_requests"],
            "c32_requests_validated": 192,
            "workload_sha256": c32["selected_workload_sha256"],
        })
    assert len({b["workload_sha256"] for b in blocks}) == 1
    comparison = {}
    for metric in ("c1_tok_s", "c1_trimmed_tok_s", "c32_warm_tok_s", "resident_warm_tok_s"):
        candidate = statistics.geometric_mean(b[metric] for b in blocks if b["candidate"])
        baseline = statistics.geometric_mean(b[metric] for b in blocks if not b["candidate"])
        comparison[metric] = {"candidate": candidate, "baseline": baseline,
                              "change_percent": 100 * (candidate / baseline - 1)}
    result = {"state": str(args.state), "candidate_flag":state["candidate_flag"],
              "blocks": blocks, "comparison": comparison,
              "notes": "Discard C32 round0; B/B reuse one process. No statistical confidence or full C32 parity claim."}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
