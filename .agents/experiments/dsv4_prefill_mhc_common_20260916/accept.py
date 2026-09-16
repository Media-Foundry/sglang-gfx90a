"""Accept only the measured large-prefill FP32/20 owner delta."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
repo = root.parents[2]
target = root / "acceptance.json"
assert not target.exists()
summary = json.loads((root / "summary.json").read_text())
assert summary["status"] == "complete" and summary["gain_pct"] >= 2
assert abs(summary["control_drift_pct"]) < 1
assert summary["live_comparisons"] == 85 * 16 * 8
assert all(q["repeat_exact_out_of16"] == [16] * 3 for q in summary["quality"].values())
assert len(summary["cross_arm_continuations"]) == 16
assert all(q["common_prefix"] == 128 and q["distinct_outputs"] == 1
           for q in summary["cross_arm_continuations"])
checks = {(t["lhs"], t["rhs"]): t for t in summary["teacher_forced"]}
assert set(checks) == {("A1", "A2"), ("A1", "B"), ("legacy", "B")}
for pair in (("A1", "A2"), ("A1", "B")):
    t = checks[pair]
    assert t["positions"] == t["top1_same"] == t["top5_records_exact"] == 1008
    assert t["max_abs_logprob"] == t["mean_abs_logprob"] == 0
    assert t["excluded_leading_nulls"] == 16
for arm in ("check", "A1", "B", "A2"):
    plan = json.loads((root / arm / "plan.json").read_text())
    assert all(hashlib.sha256((repo / p).read_bytes()).hexdigest() == h
               for p, h in plan["sources"].items())
    assert not json.loads((root / arm / f"P32-common-mhc-{arm}.stop.json").read_text())["remaining"]
for arm, paths in summary["timing_paths"].items():
    assert not paths["legacy_splitk"]
    assert paths["premix_owner_ranks"] == (list(map(str, range(8))) if arm == "B" else [])
assert (root / "validated-launcher.sh").read_bytes() == (root / "B/start-ar-matrix.sh").read_bytes()
record = dict(
    status="validated_large_prefill_common_fp32_20_owner",
    summary_sha256=hashlib.sha256((root / "summary.json").read_bytes()).hexdigest(),
    launcher_sha256=hashlib.sha256((root / "validated-launcher.sh").read_bytes()).hexdigest(),
    scope="Original V4 TP8 C16x32K, native AR, original checkpoint, 1M logical KV",
    gain_pct=summary["gain_pct"], candidate_input_tok_s=summary["candidate_input_tok_s"],
    default_promoted=False, common_fp32_20_large_prefill_validated=True,
    universal_batch_invariance_claimed=False, small_prefill_decode_unchanged=True,
    legacy_fp16_8_comparison=checks[("legacy", "B")],
    caveat="Legacy FP16/8 is a different numerical contract, not the exactness reference.",
)
target.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
