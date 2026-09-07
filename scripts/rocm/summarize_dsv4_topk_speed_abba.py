"""Summarize independent TP4 ABBA services; discard each workload's warmup."""
import argparse
import json
from pathlib import Path
import statistics


def stats(values):
    ordered = sorted(values)
    trim = max(1, len(ordered) // 10) if len(ordered) >= 5 else 0
    return dict(n=len(values), median=statistics.median(values),
                trimmed_mean=statistics.mean(ordered[trim:-trim] if trim else ordered),
                minimum=min(values), maximum=max(values))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    arms = {}
    data = {}
    for arm in ("A1", "B1", "B2", "A2"):
        c1 = json.loads((args.directory / f"{arm}_c1.json").read_text())
        p1 = json.loads((args.directory / f"{arm}_prefill_c1.json").read_text())
        p16 = json.loads((args.directory / f"{arm}_prefill_c16.json").read_text())
        data[arm] = (c1, p1, p16)
        rows = [r for r in c1["measurements"] if r["rep"] >= 0]
        arms[arm] = dict(
            c1_http_tok_s=stats([r["tok_s"] for r in rows]),
            c1_samples=[{k: r[k] for k in ("case", "rep", "tok_s", "wall_s", "sha256", "cached_tokens")} for r in rows],
            c1_case_medians=c1["medians"],
            c1_all_cache_misses=all(r["cached_tokens"] == 0 for r in rows),
            prefill_c1_ttft_s=stats([r["prefill_wall_s"] for r in p1["rounds"][1:]]),
            prefill_c16_input_tok_s=stats([r["aggregate_input_tok_s"] for r in p16["rounds"][1:]]),
            prefill_c1_first_token_exact=p1["cross_round_completion_exact"],
            prefill_c16_first_token_exact=p16["cross_round_completion_exact"],
            prefill_c1_trials=[{k: r[k] for k in ("round", "prefill_wall_s", "aggregate_input_tok_s", "completion_sha256")} for r in p1["rounds"]],
            prefill_c16_trials=[{k: r[k] for k in ("round", "prefill_wall_s", "aggregate_input_tok_s", "completion_sha256")} for r in p16["rounds"]],
        )
    variants = {}
    for variant, selected in (("A", ("A1", "A2")), ("B", ("B1", "B2"))):
        variants[variant] = dict(
            c1_http_tok_s=stats([r["tok_s"] for a in selected for r in data[a][0]["measurements"] if r["rep"] >= 0]),
            prefill_c1_ttft_s=stats([r["prefill_wall_s"] for a in selected for r in data[a][1]["rounds"][1:]]),
            prefill_c16_input_tok_s=stats([r["aggregate_input_tok_s"] for a in selected for r in data[a][2]["rounds"][1:]]),
        )
    correct = {}
    for case in data["B1"][0]["medians"]:
        ids = [r["output_ids"] for a in ("B1", "B2") for r in data[a][0]["measurements"] if r["case"] == case]
        correct[case] = all(row == ids[0] for row in ids)
    teacher_exact = all(
        all(a[k] == b[k] for k in ("input_ids", "input_token_logprobs", "output_top_logprobs", "output_ids"))
        for a, b in zip(data["B1"][0]["teacher_forced"], data["B2"][0]["teacher_forced"]))
    summary = dict(
        scope="A=legacy Top-K, B=deterministic Top-K; both keep cache/shuffle fixes; no scheduler overlap; native TP4",
        arms=arms, variants=variants,
        B_vs_A_median_change_percent={k: 100 * (variants["B"][k]["median"] / variants["A"][k]["median"] - 1)
                                      for k in variants["A"]},
        B_cross_process_c1_output_exact=correct, B_cross_process_teacher_forced_exact=teacher_exact,
    )
    encoded = json.dumps(summary, indent=2)
    if args.output:
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
