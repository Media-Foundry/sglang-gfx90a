# Wider C4 query reuse: component evidence, no production guard change

Baseline production remains original V4 TP8/native AR C16x8K, about6959
input tok/s with query16/runtime-M and pre-mix8. This independent coverage
experiment does not change that8K number or any production selector.

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

Next add M8192/16384/65536 coverage before a default-off wider selector.
Then generate16 distinct real16K/32K code-review prompts using the same
frozen public source revision as the8K workload (`54b93c45c2`), not random
text or repeated filler. `build_inputs.py` is prepared but has not yet run
at this checkpoint. Runtime code remains current; git is used here only to
freeze benchmark text with source hashes and public URLs.

Serving gates still outstanding: actual wide-backend hits, unchanged input
IDs and full KV selection,1M pool retained, real prefix-hit metadata, long
output comparison and same-input ABBA. Do not merely remove width guard or
claim the component's1.9x as E2E gain. No model weights/precision changed.
