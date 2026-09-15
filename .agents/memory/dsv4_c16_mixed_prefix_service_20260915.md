# Mixed-prefix wide-query service trial: completed performance/quality review

Chronological pending notes below are superseded by the final closure section.

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

## A1 formal timing complete; quality/candidate still pending

Three newly-computed-token rates:4114.456887,4098.389140,4096.273974 tok/s;
median4098.389140. Actual cache vectors remain exactly the planned pattern
in all three rounds. Full-input rate at the median wave is6547.207558 tok/s,
not a zero-cache comparison. Priming times36.4735/36.4839/36.4869s are excluded.
The owned A1 service continues into two128-token quality waves; do not stop
or restart it based on these partial notes. B and A2 remain required forABBA.
Formal throughput range/median is0.443660%; no recorded slow Triton compile
warnings in the formal log interval. First-token matches between formal wave
pairs(0,1)/(0,2)/(1,2) are13/16,12/16,12/16: stable timing does not imply
stable greedy outputs, even with identical IDs and cache-hit counts.

## A1 quality complete; candidate B running

A1 completed and stopped with remaining owned PIDs=[]; B PID1516348 was
birth-verified live and entered warmup. Always revalidate live state.
Read both A1 quality waves (all16 first excerpts,15 changed second excerpts).
32/32 input echoes and output-token decoding verified, counts128 each,
actual cache vectors identical. Full-output repeat1/16, first-token repeat12/16.
No obvious repetition collapse/garbling, but not a general accuracy pass:
case8 first wave asserts missing distributed helper code, contradicted by
definitions in the actual decoded input. Second wave discusses the functions.
See `manual-review.md` for bounds and case-specific evidence. This control
variation must not be attributed to the wide-query candidate, which was off.

The prior config20 trial on zero-prefix32K did not establish global drift
improvement. A future mixed-prefix config20/refinement check would be a new
workload test, not an already-proven determinism fix. Do not silently enable
it inside the current wide-query ABBA; its measured source/config remains frozen.

## Candidate warmup (formal result still pending)

B warmup newly-computed throughput4962.564401 tok/s, wave66.1331468s;
328190 newly computed tokens, the same cached vector and16/16 identical full
input echoes versus A1 warmup. All eight ranks log wide-query16/runtime-M1
selected, with the first-hit C4 capacity4096 during priming. This first-hit
log is NOT a per-forward proof that every later capacity8192 call hits the
path. No additional probes or source changes occur during the service trial.
B1/B2 formal waves and final A2 remain required; do not promote on warmup.

First B1 formal wave:4979.736918 newly-computed tok/s, wave65.9050880s,
full-input rate7955.167281 (includes196096 cached tokens; not zero-prefix
throughput). Actual cached vector matches A1 exactly. This single candidate
wave is about21.5% above A1 median4098.389140, not yet a finalABBA result.
`partial-B1-first-wave.json` independently rechecks completed-wave input echoes
and cache quantities and retains first-token comparisons; it is explicitly
partial and must not be treated as completed quality or drift attribution.

## B1 formal leg complete

Rates4979.736918/4993.264899/4979.359971, median4979.736918 newly-computed
tok/s. All actual cached vectors remain identical. B2 has started; quality
and final-control review are outstanding. This is about+21.50% versus A1,
not the finalABBA center.

Analysis now independently recomputes `max(first)-min(begin)` from all16 raw
request timestamps, validates finite ordered times and both throughput
numerators, and keeps priming separate. Four synthetic CPU tests reject
drain-time substitution, missing/nonfinite/reversed timestamps and counting
cached tokens as newly computed work. These checks pass on all completed
A1 warm/formal/quality and B warm/B1 waves; no new GPU requests were issued.

## B2 timing complete; candidate quality still running

B2 rates4972.742978/4976.427172/4952.234779, median4972.742978. Both B legs
are complete, matching cache vectors and raw timestamp accounting. No slow
Triton compile warnings in either formal interval. Candidate center of the
two leg medians4976.239948 newly-computed tok/s; final A2 is still required.
No throughput promotion or candidate quality verdict yet.

Do not compare this remaining-token rate directly to zero-prefix32K throughput:
the retained queries have longer average visible history after prefix hits.
This is a workload distinction, not a measured per-stage attribution; a
mixed-prefix critical-path profile would be needed to quantify the causes.

## Candidate quality complete; A2 loading

Read all32 B quality excerpts, independently verified32/32 input echoes and
tokenizer decoding,128 output IDs each and identical planned/actual cache
vectors. No obvious looping/garbling; claimed code defects remain unverified.
Full-output repeat0/16, first-token repeat12/16, versus A1's1/16 and12/16.
Both candidate waves discuss the actual distributed helper code in case8.
Do not infer global determinism, quality improvement, or causal drift
attribution from these unstable-control results. See manual-review.md.

B's owned process tree stopped with remaining=[]; A2 PID1527919 was
birth-verified live and loading. Revalidate before further work. A2 timings,
quality and final analysis/package are still outstanding; defaults unchanged.

## All formal timing legs complete; A2 quality pending

A2 rates4103.358391/4104.523344/4070.430258, median4103.358391. Complete
timing ABBA centers: **4100.873765 ->4976.239948 newly-computed tok/s,
+21.345846%**. This is C16x32K mixed0/25/50/75% prefixes, not zero-prefix8K.
Independently rechecked all12 formal waves:192/192 full input echoes match,
all actual cached vectors match,328190 newly computed/524286 full inputtokens,
raw TTFT/rate arithmetic valid, same measured source hashes acrossarms. No
recorded slow Triton compile warnings in any formal log interval.

A2 remains birth-verified live and is running its two128-token quality waves.
The full analyzer intentionally still requires all arms complete and stopped.
Thus performance timing is complete, but acceptance, final quality comparison,
evidence packaging and default-policy decision are not complete.

## Final closure

Sweep exited0. A1/B/A2 all completed and their owned process trees stopped;
amd-smi reported no remaining GPU owners. Full analyzer passed, including
96/96 quality input echoes, tokenizer decoding, source/config identity,
1M pool, actual cache quantities and raw timestamp arithmetic. Bounded review
of all candidate excerpts and unique controls completed. No obvious repetitive
collapse/garbling, but factual answer limitations are retained in manual-review
(especially control case8); this is not an accuracy or determinism certificate.

Quality full repeats A1/B/A2=1/0/2 of16, first-token repeats12/16 in each.
Input IDs and cache counts match; equal cache values are not established.
The low control repeatability prevents causal drift attribution from whole
generation comparisons. No claim that wide-query or config20 fixes global drift.

Qualified mixed-prefix performance result:4100.873765 ->4976.239948 newly-
computed input tok/s, **+21.345846%**, with all12 formal waves cache-matched.
The accepted8K zero-prefix6959.4558 checkpoint is unchanged. Wide-query stays
**default-off pending cold-shape closure**, not rolled back;16K/32K zero-prefix
and this mixed-prefix coverage now have service evidence.

Evidence archive `service-evidence.tar.gz`:105 files,33655184 bytes,
SHA256 `687e3e91028c1349e05899842eaf24c86acf7bb6621aec7e82abd8549fd3ade9`.
Includes raw inputs/responses/logs, scripts, critical source snapshots, summary
and explicit bounded-review record. Final summary and review are in the same
experiment directory. Only after all services stopped, started the independent
pre-mix column-pair component on physicalGPU4; no service timing overlap.
