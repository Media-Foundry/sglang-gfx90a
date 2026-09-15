# Mixed-prefix wide-query service trial: active, no accepted gain yet

Follows the completed zero-prefix wide-C4 C16x16K/32K tests. Original V4,
TP8/EP1/native AR/no-A2A, original weights,1M logical KV,32768 prefill budget.
Scope is sixteen fixed32K real-code requests with0/25/50/75% exact prefixes.
All arms retain query16/runtime-M/mix8; config20/refinement remain off. Only
wide-query changes. A1 -> B1/B2 -> A2, three formal waves/leg, one warmup per
fresh process, then two separate128-token quality waves.

Evidence/driver directory:
`.agents/experiments/dsv4_c16_mixed_prefix_service_20260915/`
`run.py`, `sweep.py`, `analyze.py`, `status.py`, `README.md`.
Client `../dsv4_c16_query_wide_20260915/bench_prefix.py` extended with an
explicit128-token quality mode; six CPU contract tests pass. Production and
runner/client hashes frozen across the trial. No other GPU jobs overlap.

Initial control A1 PID1506706 was birth-verified live; re-check `status.py`
rather than assuming this PID remains live. Sweep handle65699. Do not restart
an arm merely because observation times out. Output dirs refuse overwrite.

## First warmup evidence (not formal or candidate performance)

All16 full input echoes match. Actual cached-token vector exactly matches
planned page-aligned prefixes:
`[0,8192,16384,24576,0,8192,16384,24576,0,8192,16128,24320,0,8192,16384,24576]`.
All12 prime calls initially have zero cache hits. Full input524286 tokens,
cached196096, newly computed328190. Full-request wave TTFT80.1343954s;
full-input rate6542.5838 versus newly-computed rate4095.4948 tok/s.
Priming51.8228221s is separate and excluded. These rates must not be confused
with zero-prefix8K/32K performance or a candidate improvement.

Logs show actual prefix hits and2-4-request admissions, unlike many one-request
32K zero-prefix forwards. This can affect MHC/GEMM numerical paths; do not
attribute output changes only to caching or to the wide-query candidate.

Formal rounds are still pending at this checkpoint. Each later wave/arm must
match the actual cached-token vector; mismatches stop the trial and preserve
raw responses. Final acceptance needs all arms complete/stopped, source/input
and tokenizer checks, all-rank dispatch, warm versus formal compile evidence,
bounded manual output review, and separate within/cross-arm drift reporting.
The default wide-query flag has not been promoted.

## Partial control drift evidence, before candidate starts

`partial-A1-two-waves.json`, produced by `audit_partial.py`, checks warmup
and the first two completed A1 formal waves. Full input echoes48/48 match,
actual cache vectors match exactly. First-token matches versus warmup are
13/16 then12/16. Changed cases include zero-hit requests12 (both waves) and4
(second wave), as well as primed requests1/9/10. Examples are headings `##`
versus `#`, or `Based` versus `Looking`; these one-token excerpts alone cannot
judge answer quality.

This is **wide-query off in all three waves**, so the candidate is not needed
for drift to occur. Changed input IDs/cache-hit quantities are excluded for
these pairs, and reusing a prefix is not necessary (zero-hit cases also drift).
Equal cache counts do not establish equal cache values. Dynamic admission,
GEMM/MHC branches and atomic MoE reduction remain possible contributors;
this partial audit does not isolate the first numerical divergence.
