# Small native-AR screen: original V4 Flash, TP8

Same original checkpoint,1M allocated KV pool, proper native graph tiers.
DSpark off. ABBA, two fixed real-code natural-EOS waves per leg.

| Concurrency | Control resident tok/s | Candidate resident tok/s | Interpretation |
|---|---:|---:|---|
| C32 |1044.61|1060.70|+1.54%, promising small gain|
| C1 |89.72|90.02|Essentially unchanged; M1 never selects the candidate|

C32 control return shifts-0.60%; B1/B2 differ0.094%. Every candidate C32 wave
exceeds every control wave. This is a small screen, not a formal broad-load
acceptance or proof of a1.54% full HTTP request gain. Full HTTP rates remain
separately recorded and vary with output length/admission/drain.

The new `SGLANG_DSV4_GFX90A_TP8_M32_DOWN_CONSUMER=1` is default-off and scoped
to native TP8/EP1 M32/I256 A4/R2. No C1, prefill, DSpark or weight changes.
C1 already quantizes inside its direct down kernel; this point checks
non-regression, not an M1 port. This fixed-case C1 screen does not supersede
the previous multi-case formal87.599tok/s result.

Synthetic component: quant+down+reduction time-11.21%/-13.26% in two routing
regimes, each with100 mutations and1000 checked graph replays; FP32 partial
and BF16 output numerically equal. Service France passes;264 measured outputs
independently checked against token IDs. C32 controls themselves remain
cross-wave nondeterministic, so whole-model bitwise equivalence is not claimed.

All experiment services are stopped and GCD0--7 have no running processes.
See `summary.json`, `C1-summary.json`, `component.json` and `README.md`.
Raw evidence is retained in `measurement-evidence.tar.gz`; its member hashes
are in `evidence-index.json`. No raw files were deleted after packing.
