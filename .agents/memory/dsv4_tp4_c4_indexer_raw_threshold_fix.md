# DSV4 TP4 C4 indexer raw-length threshold fix

Date: 2026-08-30

## Root cause

`index_topk=512` is measured in C4-compressed cache rows. Both the C4 skip
predicate and decode dual-graph dispatcher compared this value directly with
full-resolution `ForwardBatch.seq_lens`. They therefore enabled the full
indexer at raw position 512, when only about 128 C4 rows existed and Top-512
necessarily selected every valid row. The correct raw-token threshold is
`index_topk * compress_ratio = 512 * 4 = 2048`.

This was not VRAM/KV eviction. The strict BS32 run used about 17--18k active
tokens in a 32768-token pool and retained roughly 11 GiB physical VRAM
headroom per GCD. There were no retractions and every request finished at the
requested length.

## Evidence

Before the fix, strict 32-request native-AR throughput in 16-generated-token
windows was approximately:

- generated 464--480: 675.8 tok/s;
- 480--496: 469.6 tok/s as requests crossed raw position 512;
- 496--512: 134.0 tok/s;
- 512--544: 154--158 tok/s.

Forcing even short contexts to capture/replay only the sparse graph produced
157--162 tok/s from generated token zero onward. This proved that the full
indexer path itself, not growing KV residency or L2 eviction, caused the
long-lived drop.

Standalone BS32/L513 indexer timings were only about 21.8 us for logits and
13.3 us for production TopK/slot transform. The much larger service cost is
partly caused by the sparse graph's static maximum C4 width, which is derived
from the full page-table capacity rather than a runtime length bucket. That
separate long-context issue remains after raw position 2048.

## Result

After comparing raw sequence length against `512*4`:

- strict BS32 resident throughput: 700.57 / 701.90 tok/s;
- pre-fix three-run center: about 628 tok/s;
- improvement: about 11.6%;
- generated 512--544 recovered to about 662--692 tok/s;
- 32/32 teacher-forced next-token IDs matched the accepted first-token oracle;
- France checks passed and every request returned 544 tokens.

There is still a transient 496--512 window at about 420--426 tok/s, followed
by full recovery. This is consistent with a one-time compressor/cache state
transition at raw position 512 and is not the old persistent sparse-indexer
path.

Completion hashes differ from the pre-fix service after the old sparse path
would have activated. The pre-fix full-indexer graph and the corrected dense
all-valid path visit C4 rows in different floating-point reduction orders;
the user-approved correctness policy accepts such small asynchronous-order
differences when teacher/France and completion-health gates pass. The fix does
not approximate TopK: below 2048 raw tokens there are at most 512 C4 rows, so
all valid rows are selected by definition.

## Remaining P0

Once raw length exceeds 2048, the sparse graph still allocates/scans logits at
`page_table.shape[1] * 64`, the full graph capacity. Proper long-context work
requires length-bucketed sparse graphs or fused scan/local-TopK; do not revert
the corrected C4/raw threshold.
