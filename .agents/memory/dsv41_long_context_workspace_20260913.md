# V4.1 long-context score workspace and real-code validation

Parent checkpoint: `3b711c8835`. Original weights, TP8/EP1/noA2A, eager,
Engram large tables stay in host RAM. This is correctness/capacity work.

## Measured bottleneck, not a guessed KV OOM

On isolated GPU4, the original score expression materializes QK, ReLU and
weighted BF16 tensors of shape[M,32,L] before the head reduction. For M2304,
measured incremental allocated peak is3.375GiB atL8192,6.75GiB atL16384 and
13.5GiB atL32768. AtL65536 it is27GiB. These are component allocation peaks,
not total serving peaks or model storage, and include output allocation.

The new `bounded_indexer_scores` tiles independent query rows with a128MiB
QK slab, retains full head/K reductions and BF16 rounding points, uses
in-place ReLU/multiply, and releases each slab before allocating the next.
The final FP32[M,L] score matrix is still materialized. This is not a fused
scan/Top-K algorithm and does not remove its remaining O(M*L) output/sort cost.

Final full-workspace guard (`workspace-guard.json`) measurements:

| M | L | Reference peak | Bounded peak | Full scores / Top512 / candidate blocks |
|---:|---:|---:|---:|---|
|2304|1152|486MiB|142.59MiB|all exact|
|2304|2304|972MiB|152.72MiB|all exact|
|2304|4608|1944MiB|172.47MiB|all exact|
|2304|8192|3456MiB|204MiB|all exact|
|2304|16384|6912MiB|276.5MiB|all exact|
|2304|32768|13824MiB|420.5MiB|all exact|

RaggedM129/L16384,L32768 and M2304/L65536 also pass. Inputs are synthetic
real-shape BF16 values round-tripped through the actual per32 FP4 fake-quant
helper, with signed BF16 head weights. They are not captured model activations.
Bench script freezes the pre-tiling reference expression rather than comparing
the candidate with itself. Component timings are a small sequential screen,
NOT an E2E/ABBA speed claim. For example L32768 increases from46.66ms to49.29ms
in the final2-round screen despite the large memory reduction.

## Failed first32K service attempt and corrected guard

Initial wiring required both a large workspace and L>8192 to preserve the
small-context path. This was too narrow. After a successful France request,
the fixed22316-token code request failed at12:31:29 in the ORIGINAL score
ReLU/multiply path while requesting648MiB (consistent with M2304/L4608).
Reported free memory was402MiB, allocated56.66GiB, reserved-unallocated5.54GiB.
The latter is not proof a contiguous648MiB allocation was available. No
permission problem, malformed API prompt, or model-quality claim explains it.

Failure evidence is retained as `initial-oom.txt` and `failed-long-request.json`.
PID113685 terminated through normal crash cleanup at12:32:34; the client got
RemoteDisconnected after80.61s. We verified PID/port/GPU state before restarting.

The final selector uses required slab bytes, not L. C1/small workspace keeps
the original method; large-M early chunks now also get bounded scores. All six
lengths in the table were retested after correcting this guard. Large short
prefill can therefore enter the new path: do NOT repeat the initial claim that
all <=8192-key calls are untouched. Mixed q/k/weight dtypes retain the original
promotion path, because BF16 in-place multiplication would otherwise be wrong.

## Runtime configuration and E2E evidence

`start_dsv41_correctness.sh` now defaults CONTEXT_LENGTH to MAX_TOTAL_TOKENS;
the common launcher forwards it only when explicitly set (old V4 defaults
unchanged). This avoids advertising the unvalidated1M model-card context on
the small test pool. The selected test profile is pool32768/context32768,
chunk2304, no graphs, same original model and RAM Engram. This is a changed
capacity configuration, not an identical-config performance comparison.

First bounded-score retry: PID123083, tmux
`dsv41-long32k-bounded-20260913`, log
`/tmp/dsv41-long32k-bounded-20260913.log`. Ready at12:37:21 HKT. The same
22316-token request now completes in95.185s, produces90completion IDs with
natural EOS, no prefix cache hit, finite/aligned logprobs, and all seven
typed factual answers correct. Full response and exact prepared input are
retained compressed under the experiment directory. This proves this long
factual smoke, not general quality or the full32768 capacity limit.

France still gives Paris with the same8 IDs. Before the long request, two
fixed source-review prefixes2376/2560 also preserve next-token agreement to
the earlier unbounded checkpoint (tokens15/14). Logs show the bounded method
actually hitting M2304/L1152 andL2304 in the full model. The long request
completes9x2304 plus1580 real input rows (last forward padded1792), crossing
the16384 candidate-pruning boundary without the earlier OOM.

Four cached-vs-full-recompute comparisons of that long answer are complete at
lengths22316/22399/22400/22401:4/4 top1 agreement. The first also has identical
top20 logprobs. Later points have top20 overlaps0.95/0.85/0.95 and maximum
common logprob deltas0.687501/0.625/0.375. This is not floating-point bitwise
parity. The22399..22401 points straddle an absolute128-token boundary; their
top1 margins are large, so this is not an adversarial small-margin oracle.
Full recompute responses are preserved in `long-prefix-recomputes.tar.gz`.

A separate31343-token real-source factual request failed at12:49:55 in
`stable_index_topk`'s full stable `argsort`, requesting790MiB. Reported free
memory292MiB, allocated56.93GiB, reserved-unallocated5.35GiB. This is a SECOND
workspace bottleneck, not a recurrence of the QK/weighting allocation. Client
received RemoteDisconnected after103.82s; PID123083 exited at12:50:00. Evidence:
`topk-oom.txt`, `31k-failed-request.json`, exact `31k-prepared-input.json.gz`.
The29464-token intermediate preparation was not executed; do not count it as
an additional passing workload.

## Stable-sort workspace bound (second fix)

`stable_index_topk` now sorts independent query-row slabs and retains only
the final[M,k] output, not the full[M,L] int64 permutation. Score order,
lowest-ID cutoff ties, and ascending emitted logical IDs remain unchanged.
Small arrays use the original single sort. The final slab bound is4194304
elements; it is an element bound, not a guaranteed backend scratch byte cap.

Isolated M2304 tie/causal fixture, original stable sort vs bounded sort:

| L | Original peak | 1M-element slab peak/time | Selected4M-element slab peak/time | Original time |
|---:|---:|---:|---:|---:|
|32768|2.25GiB|53MiB /61.12ms|137.0MiB /17.63ms|12.22ms|
|65536|4.50GiB|53MiB /122.65ms|185.5MiB /29.88ms|23.17ms|

Both slab choices preserve every selected ID, pass100graph replays per shape
and10score/position mutations. The initial1M choice also passed the smaller
CPU-reference Top-K fixtures with1000graph replays each. The4M choice avoids
most of the1M launch overhead while keeping allocations far below the failed
790MiB request. These are component screens, not service speed gains.

After a test-only positional `argsort` argument was corrected to `dim=-1`,
the full suite passes90tests+16subtests. Production already used `dim=-1`;
the three initial unit failures were in the test oracle, not service code.
When rerunning after context handoff, two plain-pytest collection attempts
failed because `sgl_kernel` was absent from that shell's Python search path
(setting SGLANG_USE_AITER alone did not fix it). The launcher uses the local
AOT build/python directories in PYTHONPATH. Use the full environment recorded
in `dsv41_candidate_blocks_20260913.md`, not a stub or a different installation.

Retry service: PID137272, tmux `dsv41-long32k-sort-20260913`, log
`/tmp/dsv41-long32k-sort-20260913.log`, same32768pool/context and2304chunk.
Ready at12:58:11 HKT. The two identical31343-token requests both complete
without OOM, with91completion tokens, natural EOS, zero cache hits/retractions,
in126.287s and124.422s. All91output IDs, all91selected-token logprobs and
all91top20 rows are exactly identical between these two runs. This is repeat
evidence for this fixed input/configuration, not arbitrary-batch determinism.

Both FAIL the original strict JSON checker: the model returns the correct
value20 under `candidate_source_layer_id` instead of the requested
`candidate_source_layer`. The other six typed fields agree exactly. Keep
`passed=false` and the unmodified prompt/checker. Capacity/no-OOM success and
this stable schema deviation are separate observations; neither proves the
cause of the schema deviation. Completion ID SHA256:
`824c7c7eaa6dcecc3a28ac1227173c6016e53ba337dad0cc062facf7b1fa1867`.

After these near-full-pool requests, C4 smoke repeated twice passes8/8:
France, arithmetic, Python slicing, SQL count all finish normally with valid
completion-only IDs. Each case repeats identical IDs across the two C4 rounds;
this is semantic/API and cache-recycling smoke, not a large-batch quality suite.
The current service's standalone France probe also passes, answering
"The capital of France is Paris." with8tokens including EOS.

Current serving `dsv41_sparse.py` SHA256:
`6a28f9a00aa18aeffdaaad90ceada738662b1828b0f486d3c92f0e810d8eb144`.
Earlier bounded-score-only service source SHA256:
`06022629cd7130538b07bb731c6ee2ffe1f0b2be7b3127b4f02ffe998617f00b`.

The prompt was prepared once by `check_dsv41_long_code.py`: four complete real
Python files plus actual checkpoint fields,22316token IDs, seven factual JSON
answers. Retry uses the same IDs as the failed run; it does not shorten the
prompt or change chunk size. The quoted source predates the guard correction,
but none of the seven queried facts changes. Source hashes are in the prepared
artifact. The checker requires complete input accounting, completion-only IDs,
finite/aligned token and top20 logprobs, exact typed factual answers and EOS.

The helper's CPU tests cover zero/one/ragged rows, empty keys, signed values,
BF16 andFP32, tiny andlarge budgets, input nonmutation and mixed-dtype rejection.
The broader bringup/API/candidate/SWA/score suite passed87tests+16subtests again
after the guard correction (12.51s). No full-service
graph, independent full-model reference, or broad quality claim is made.
