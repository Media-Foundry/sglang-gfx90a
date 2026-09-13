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

## Runtime configuration and pending E2E

`start_dsv41_correctness.sh` now defaults CONTEXT_LENGTH to MAX_TOTAL_TOKENS;
the common launcher forwards it only when explicitly set (old V4 defaults
unchanged). This avoids advertising the unvalidated1M model-card context on
the small test pool. The selected test profile is pool32768/context32768,
chunk2304, no graphs, same original model and RAM Engram. This is a changed
capacity configuration, not an identical-config performance comparison.

Current retry at recording time: PID123083, tmux
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

Four cached-vs-full-recompute comparisons of that long answer are now running
at lengths22316/22399/22400/22401; results are not yet claimed. The22399..22401
positions straddle an absolute128-token boundary. A longer code prompt is
being prepared separately; it is not a completed test.
Serving `dsv41_sparse.py` SHA256:
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
