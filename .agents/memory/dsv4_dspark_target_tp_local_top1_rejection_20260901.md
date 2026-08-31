# DSpark target TP-local greedy top-1 oracle (2026-09-01)

## Question

For greedy target verification at gamma three / M128, can each TP4 rank take
its local LM-head argmax and exchange only `(score, global_token_id)`, instead
of gathering the complete 129,280-token vocabulary and then applying argmax?
This is exact for the all-greedy/no-adjustment case when ties select the lowest
global token ID, and can be guarded entirely inside target verification.

## Oracle

`scripts/rocm/bench_dsv4_dspark_target_tp_local_top1.py` ran on physical GCDs
4,5,6,7 with 128 rows and four 32,320-token vocabulary shards. The candidate
used a MAX reduction for the BF16 local scores followed by a MIN reduction of
the winning global IDs. IDs were transported as FP32; every DSV4 vocabulary ID
is below 2^24 and therefore exact.

Across 100 activation mutations, including forced equal-score cases, the
candidate and gathered-vocabulary reference had zero token-ID mismatches on
all four ranks.

Seven alternating-order rank-max samples gave:

| arm | median | trimmed mean |
|---|---:|---:|
| gather full vocab + global argmax | 777.647 us | 777.850 us |
| local argmax + score/ID reductions | 687.689 us | 686.818 us |

The exact saving is about 91.0 us per target step (11.7% of this isolated
tail), but below 0.2% of the measured 60--70 ms gamma-three service step.

## Decision

Do not widen `LogitsProcessorOutput` or the DSpark epilogue for this isolated
gain. Retain the standalone oracle as a lower-risk reference if a future
fused LM-head epilogue can emit local top-1 without materializing local logits.
No AR or production path was changed.
