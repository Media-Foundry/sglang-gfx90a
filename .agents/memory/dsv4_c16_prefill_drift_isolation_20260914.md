# C16 8K drift: partial atomic repair, remaining batch dependence

2026-09-14, base `e0aea46b77`, original V4 Flash TP8/EP1/no-A2A,
chunk32768, 1M allocated KV, native AR. Evidence:
`.agents/experiments/dsv4_ck_drift_20260914/`.

**Do not claim the full drift problem is solved.**

## Confirmed component source and optional repair

Frozen CK stage-1/sorter/weights with random M8192 routes: original FP32 atomic
stage2 **0/30 byte-exact**, fixed-slot **30/30**, and100 graph replays plus5
input mutations exact (`m8192-bitwise.json`). Original replay can change a final
BF16 element. Larger regular/random M32768 fixtures sometimes reproduce exactly
even with atomics; nondeterminism depends on contention/timing, not just a flag.
Repeated full-stage sorter and stage1 snapshots are stable in these fixtures.

New `SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=1`, **default0**:
remap token/slot into M*6 virtual tokens, call unchanged CK stage2 with TopK1,
then fixed-order FP32 slot sum in HIP and BF16 cast. Preserves router weighting,
stage1 and packed-weight semantics, but not old atomic reduction order.
M32768 scratch3GiB/GCD vs0.5GiB original. Real TP8 service still fits1M pool.
No new weights, no CK header/binary rebuild, no small/decode kernel change.

After fixed reduction, synthetic full CK row permutation and M->M+1 preserve
all retained outputs at M8192 and32768. This is not a real-layer/full-model oracle.

## Main service finding

Same16 real8K code inputs, fixed128 output length, fresh salts:

| Path | Ordered batch outputs, 3 pair comparisons | Ordered logprobs | Normal independent HTTP requests |
| --- | --- | --- | --- |
| atomic |16/16 each|15/16,16/16,15/16|5/16 outputs repeat|
| fixed slot |16/16 each|16/16 each|5/16 outputs repeat|

Initial normal harness did not request logprobs, unlike ordered batches.
**Explicit confound check completed in a fresh atomic service:** both protocols
request identical logprobs/top5 and sampling options. Ordered batch repeats
remain16/16; independent requests repeat only3/16. `forward_entry_time` proves
changing group membership for independent requests and stable0..3/4..7/8..11/
12..15 groups for ordered batches. No cached tokens in either protocol.

Thus there is strong evidence for batch-composition/row-position numerical
sensitivity, not just random atomic updates or the logprob option. Exact first
offending model operator for this8K fixture remains unlocated. Do not call
ordered admission a production batch-invariance fix or blame scheduler correctness.

Atomic vs fixed controlled full128 outputs match0/16; fixed order changes
arithmetic. First divergences include ties and non-ties. Inspected candidate
prose has no obvious collapse; France passes all fresh services. These are
bounded smoke checks, not universal semantic or bitwise correctness.

P diagnostic A/B:5296.66 atomic vs5101.91 fixed input tok/s (-3.68%).
Not ABBA acceptance; **do not enable fixed-slot by default**. Preserve fast
32768 C16 profile, keep this option as a numerical oracle. No claim of speed gain.

## Next useful direction

Existing `dsv4_projection_experiment.maybe_row_stable_linear` only covers
M5..4096, and ROW_STABLE_PREFILL defaults0. It cannot protect current32K prefill.
Read `dsv4_row_stable_prefill_diagnostic_20260908.md`: historical first differences
were Q, wo_a, compressor and later router, not a broken cache; that small fixture
was fixed, while full AR remained batch-sensitive. Reuse its stage capture and
test large-M row/shape invariance before broadening the guard. Do not rerun
BF16 CAS, assume all drift is CK sorting, or infer whole-model determinism from
a graph replay/final output hash.
