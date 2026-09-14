# Wider C4 query reuse: component evidence and default-off wrapper

Baseline production remains original V4 TP8/native AR C16x8K, about6959
input tok/s with query16/runtime-M and pre-mix8. This independent coverage
experiment does not change that8K number. A separate, default-off wider
wrapper admission has now been added; service acceptance is still pending.

## First four fixtures passed

Current empty-tile logits + unchanged deterministic Top-K compared with the
unchanged runtime-M query16 kernel called directly beyond the wrapper's
width2048 limit. The production wrapper is explicitly checked to decline
each wider fixture. One physical GPU4 only, after checking all GPUs free.

| Synthetic fixture | M | C4 width | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|---:|---:|
| Two16K causal requests |32768|4096|65.9849|34.0357|1.9387x|
| One32K causal request |32768|8192|133.8045|70.3109|1.9030x|
| Mixed prefixes0/4096/16384/24574 |32768|8192|127.1218|68.0298|1.8686x|
| Odd M, partial final tile, unrelated pages |17|8201|0.18808|0.18932|0.9935x|

Three ABBA cycles, five graph replays per event sample. Times include logits
and logical/physical Top-K, not query projection/compressor or whole service.
400 mutation checks plus100 fixed graph replays per fixture passed exact
FP32 score bits and both index outputs. Mutations change Q, weights, cache
values and page mappings; ragged mode changes lengths too. Zero-Q checks
also independently verify tie membership, descending logical emission and
physical-page translation. Larger fixtures preserve synthetic causal/prefix
lengths during mutation; these are not live service metadata captures.

The wider large-M candidate uses64 registers,0 spills,8KiB LDS. Width8192
requires1GiB FP32 logits per arm (2GiB for both oracle outputs), not extra
production workspace beyond the existing dense-logit representation. Mixed
prefix rows are8191/8193/8190/8194 and the page-table stride is131, exercising
request boundaries and non-contiguous row stride.

CPU initial-geometry accounting reports nonempty group tiles, shared-page
fraction and logical K tile-load reuse. Mixed-prefix initial shared-page
fraction is0.99909; logical K-load reuse15.78x. These are synthetic metadata
counts, NOT measured HBM savings or kernel speedup. Four CPU tests cover the
accounting and tie rules.

## Failed initial oracle, retained

The first `screen.json` is incomplete: my additional independent tie check
incorrectly expected ascending emission. The production contract selects
smaller logical IDs on equal scores but emits selected IDs in descending
order; the length<=K shortcut also emits descending IDs with -1 padding.
The control/candidate equality checks had passed, but this extra assertion
failed. Corrected the oracle, not the production Top-K. Preserved the original
driver as `failed_screen_source.py` and its log/result. The corrected v2 and
full runs completed. This was an agent-created test error, not a GPU fault.

## Reproduction and remaining gates

Directory `.agents/experiments/dsv4_c16_query_wide_20260915/`.
First-suite archive11 files,10156 bytes; SHA256
`07a13e6b4bb2413b4e00ae157b3c4ec4d5491b28d6a70df5033de9bf1cfdb667`.

Supplemental M8192/16384/65536 coverage is complete: another400 mutations
and100 fixed graph replays per fixture passed. Full logits+Top-K times:

| Fixture | Control ms | Candidate ms | Speedup |
|---|---:|---:|---:|
| M16384,16K causal |32.99364|17.04870|1.93526x|
| M8192,8K prefix +8K queries |23.43623|12.23005|1.91628x|
| M8192,24K prefix +8K queries |53.55555|28.00165|1.91259x|
| M65536,two32K requests |267.57317|140.41664|1.90557x|

The actual wrapper now accepts `allow_wide=True` only with query16,
runtime-M, M8192..65536 and width<=8192. Default remains False. The outer
original-V4 TP8 ordinary-EXTEND guard is unchanged. Indexer exposes the
independent `SGLANG_DSV4_C4_PREFILL_QUERY_WIDE=1` flag, with no launcher
default. A separate per-rank wide-hit log avoids losing this evidence if
narrow prefix priming triggers the existing first-hit message first.

Integrated wrapper: mixed-prefix, M32763/W4096 and M32767/W8192 all passed
another300 mutations plus100 fixed graph replays each. Full-stage speedups
were1.87696x,1.94115x and1.90314x respectively. These are component figures,
not whole-indexer or E2E speedups. CPU suite:29 tests,56 subtests passed,
with one pre-existing unknown asyncio_mode pytest-option warning.

Real manifests are complete:16 distinct code reviews per length, using
the frozen public source revision `54b93c45c2`. 16K has262141 total tokens;
32K has524286. All32 requests were re-encoded through the official chat
encoder and matched saved input IDs and hashes. The JSON `prompt` field is
user content, not the full formatted chat string: an initial ad-hoc raw
tokenizer check returned False because it omitted that chat template;
`validate_inputs.py` uses the correct encoder and passed. No input data
were changed to make the validation pass.

Integrated-evidence archive15 files,1827323 bytes, SHA256
`896be4c928c8a9b014612acd6ce1bb3a73aa1013a95c8d5f66fe302f1837bd93`.
It includes the integrated driver, supplemental run's earlier driver hash,
results/logs and both real manifests. Runtime code remains current; git is
used only to freeze benchmark text with source hashes and public URLs.

A fresh C16x16K A1/B1/B2/A2 service sweep is now launched under
`.agents/experiments/dsv4_c16_query_wide_service_20260915/`. Only the wide
flag changes; query16/runtime-M, MHC8/post-fused4,32K chunk and1M KV stay
fixed. Each leg has3 measured waves and each service two128-token quality
waves; France is checked before long input. No service result is claimed
at this checkpoint. The driver checks actual candidate wide hits on all
eight ranks and refuses wide hits in the control, captures source hashes,
and stops only owned PID/birth-matched services.

Serving gates still outstanding: actual wide-backend hits, unchanged input
IDs and full KV selection,1M pool retained, real prefix-hit metadata, long
output comparison and same-input ABBA. Do not merely remove width guard or
claim the component's1.9x as E2E gain. No model weights/precision changed.

## In-progress16K service: control A1 completed

A1 warmup5361.7672 input tok/s; timed5654.7166/5655.0734/5654.8707,
median5654.8707. All24 timed forwards reported2 requests/M32768; no serving
Triton compile warnings. France returned Paris. Service stopped with no
remaining owned workers; candidate B is running, A2 not yet measured.
These are partial results, not a completed ABBA or a candidate speed claim.

Control A1's two128-token quality waves have32/32 exact full input echoes,
but only10/16 complete token sequences repeat. First divergence positions
(zero-based) are case3:1, case4:92, case8:19, case11:0, case13:32, case15:53.
All16 quality prefill forwards also report2 requests/M32768, so there is no
observed change in this coarse M histogram. This does not prove identical
row placement, physical pages, kernels' arrival order or atomic order.
Manually inspected cases3/8/11 remain coherent and discuss the corresponding
allocation/distributed-helper/FP8-scale excerpts; this is not a proof of
semantic accuracy. Do not attribute future A/B text differences to query
reuse without considering this control drift floor.

Prepared `bench_prefix.py` (not yet sent to the GPU service) for mixed
0/25/50/75-percent, page-aligned priming of exact real input-ID prefixes.
It keeps full requests unchanged, records actual cached counts and both
full-input/newly-computed-token rates, and excludes priming time explicitly.
It requires positive hits for primed rows and preserves raw responses before
assertions. Four CPU contract tests passed, including changed-input, wrong
request ID, bad cache count and empty-output rejection. A32K plan-only run
was generated; real cache/API behavior remains to be validated after the
current zero-prefix sweep. No additional requests were sent concurrently
with the timed service experiment.

## Candidate B completed; A2 pending

B1 rates6326.6799/6322.6787/6320.7745, median6322.6787 input tok/s.
B2 rates6321.2493/6324.9873/6319.5982, median6321.2493. B1 timed shape
counts match A1:24 forwards of2 requests/M32768, no serving compile warning.
The approximate11.8-percent gain versus A1 is preliminary until A2 closes.
All8 ranks logged wide-query selection at M32768/C4width4096/group16/runtime1.

B warmup was5000.9397, with a first `reuse_runtime_m` compile reported as
8.40..8.63 seconds across the8 ranks. These concurrent rank timings must not
be summed into68 seconds. Runtime-M removes repeated M specialization but
does not remove the first compile for new width/layout specialization.
Wide-context cold prewarm remains an outstanding production gate.

B quality:32/32 full input echoes exact,13/16 within-arm token sequences
repeat; differing cases2/8/13. A1-to-B across both wave pairs matches12/12/
13/12 of16. Both complete candidate waves were visually inspected (all16
first-wave excerpts plus the three changed second-wave excerpts): coherent
code-review prose, no obvious repetition collapse. Some asserted defects
are speculative; this is NOT a validation that every claimed source-code
bug is real. Control A1 itself repeats10/16, so these counts do not identify
a new candidate-induced drift cause or establish global determinism.
B stopped cleanly. A2 is loading; no final ABBA result is claimed here.

Prepared a separate32K suite in
`.agents/experiments/dsv4_c16_query_wide32k_service_20260915/`, not launched.
It keeps1M KV/chunk32768 and requires the completed16K summary to improve
over2 percent plus an explicitly confirmed bounded output review. Both
16K/32K suites have a review tool whose default is *not* to assert a semantic
smoke pass. The review/packaging tools preserve input hashes, all candidate
texts and matches against all four control quality waves.
