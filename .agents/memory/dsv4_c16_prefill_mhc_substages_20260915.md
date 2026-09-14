# TP8 C16 prefill: MHC substages and exact post reuse (2026-09-15)

## Scope and unchanged production result

Original V4-Flash, original weights, TP8/EP1/native AR, 1M logical KV pool,
32K admission budget, 16 diverse ~8K code requests with explicit token IDs.
The accepted E2E result remains **5569.784 input tok/s**, +5.1269% over the
matched empty-tile-disabled control. No new kernel selector is enabled by
this entry. V4.1, DSpark, TP4, and decode are not performance claims here.

## Complete detailed-marker service run

Building on `b412b36491`, extend the default-off async matrix from 44x32 to
44x64 uint64. Existing layer-entry and attention-output markers select the
current layer and attention/FFN boundary through a ContextVar. New slots:

- 32..36: attention-side fused MHC entry/post/mix/Sinkhorn/weighted-Norm.
- 40..44: FFN-side equivalents.
- 48..54: indexer entry/query start/weights/query/compressor/logits/Top-K.

Selected implementation names are saved alongside raw ticks. No synchronizing
collective or scheduler-thread GPU wait is added; context tokens reset even
on exceptions. Detail hooks import only when the debug directory is explicitly
set. Existing numerical expressions are untouched.

Run `markers-B-detailed` completed one warmup and three measured diagnostic
waves, all 128 rank/frame snapshots and all 64 exact input-ID echoes. No prefix
cache hits, one output token/request. All 43 coarse stages are valid. The first
layer's attention pre has no preceding fused post, so its detailed fused slots
are intentionally absent. Per wave there are 168 detailed attention MHC
boundaries, 172 FFN boundaries, and 84 C4 indexers. As before, keep every stage
from the same longest-envelope rank for a given forward, rather than summing
independent rank maxima. Absolute cross-rank clocks are not aligned.

Warm event/realtime ratio range: 0.99999731..1.00002214. Maximum within-frame
rank envelope spread: 0.68704 ms. Mean summed forward envelope: **23.4889 s**.
Diagnostic HTTP waves: 23.58882 / 23.57917 / 23.58227 s. These numbers are
not an additional A/B throughput improvement.

| Nested span | Mean seconds/wave | Approximate envelope share |
|---|---:|---:|
| MHC FP32 pre-mix, both boundaries | 4.9572 | 21.1% |
| MHC post-combine + RMS partials | 2.8128 | 12.0% |
| MHC weighted-sum + Norm | 0.5708 | 2.4% |
| MHC Sinkhorn | 0.0614 | 0.3% |
| Indexer query projection/transforms | 0.6189 | 2.6% |
| Indexer weights projection | 0.0279 | 0.1% |
| Indexer compressor | 0.1165 | 0.5% |
| Indexer logits + adjacent metadata | 2.6384 | 11.2% |
| Indexer Top-K + adjacent metadata | 0.1162 | 0.5% |

These are nested inside the earlier coarse spans and must not be added to them.
Runtime signatures confirm `post_combine_rms`, `fp32_from_partials`, and
20-iteration Sinkhorn, not the historical TP4 BF16-GEMM MHC path. A concrete
layer-2 example has ~8.30 ms post and ~14.70 ms pre-mix per boundary; its C4
logits span is ~31.56 ms. This is stronger evidence for the next optimization
than assuming all attention-prepare time belongs to indexer query projection.

## FP32 GEMM screen: defer, not an accepted fix

`screen_fp32_mhc.py` compares the actual FP32-from-partials pre-mix against
BF16 activation promotion + FP32 torch.mm + RMS scaling. Fn remains the real
FP32 layer-0 checkpoint-derived weight. Activation rows are repeated/scaled
from 19 real sampled rows: synthetic occupancy, not 32K independent real rows.
The promotion and scale kernel are included in timing; the extra promoted
activation workspace is explicit.

Three ABBA cycles, single otherwise-idle GCD4:

| M | Control ms | Candidate ms | Speedup | Extra FP32 activation workspace |
|---|---:|---:|---:|---:|
| 8192 | 3.7402 | 3.4952 | 1.070x | 512 MiB |
| 32768 | 14.8741 | 12.5726 | 1.183x | 2 GiB |

Ten fixed-input replays were exact within the candidate. Three mutations/shape
remained finite; candidate/control relative L2 was ~1.9e-6. But row permutation
changed candidate output after inverse permutation (~4.8e-7 relative L2).
There is no E2E quality or capacity acceptance for this path. It is not wired
into service; prefer the exact no-extra-workspace candidate below first.

## Four-output post-combine reuse: positive isolated result

The September-10 M128 record already found an exact four-output variant, but
its few-microsecond decode benefit did not justify wiring. The new large-M
profile gives this same work decomposition a different budget. The old /tmp
prototype was not available; `screen_post_fused4.py` reconstructs it with
H256 and four wave64s, retaining the production accumulation expression and
BF16-round-before-RMS rule. One CTA loads x and four residual channels once,
then emits all four output channels and their independent RMS partials.

### Initial implementation error, not numerical drift

The first screen used `comb[output_channel,input_channel]` rather than the
production `comb[input_channel,output_channel]` contract. Its widespread
output mismatches and apparent ~2x timing are an **invalid candidate**, not an
acceptable numerical perturbation. `post-fused4-screen.{json,log}` preserves
this failure. Correcting the addresses to base `token*16+output_channel`, with
input offsets 0/4/8/12, removes it. An added asymmetric integer-valued fixture
expects [151,166,181,196] and exact RMS partials, explicitly detecting this
transpose mistake.

### Corrected validation and timing

Real sampled layer-0 attention output/residual/post/comb are repeated/scaled;
each mutation changes all four inputs. Control is the current
`mhc_post_combine_rms_triton`, not a hand-written semantic oracle alone.
Each shape passed **100 mutations with bitwise-identical BF16 output and FP32
RMS partials**. M128 additionally passed 1000 HIP graph replays. All tests ran
only on GCD4, after the TP8 diagnostic service had stopped.

Three ABBA cycles (median of six samples/arm, five launches/sample):

| M | Control ms | Corrected candidate ms | Speedup |
|---|---:|---:|---:|
| 128 | 0.05994 | 0.03120 | 1.921x |
| 8192 | 2.05687 | 1.01672 | 2.023x |
| 32767 | 8.25376 | 4.03696 | 2.045x |
| 32768 | 8.25146 | 4.03514 | 2.045x |

The M128 eager measurement includes Python/allocation overhead and is not a
new decode performance comparison. The intended workload is large prefill.
Both paths produce the same output/scratch sizes; no added workspace or weight
copy is needed. This is **component acceptance only**. It does not establish
whole-model determinism or an E2E gain. The next step is a narrow original-V4
TP8 native-prefill selector, then real C16 service ABBA and long-output checks.
Keep decode, speculative, and V4.1 unreachable; do not default-enable from
the micro result. Preserve 1M KV and test the irregular M rows actually admitted.

## Evidence and next direction

Directory: `.agents/experiments/dsv4_tp8_c16_empty_tiles_20260914/`.
`capture_markers.py --label B-detailed` deliberately requires a fresh output
directory. `analyze_markers.py --label B-detailed` checks the completed run.
`post-fused4-full.json` is the corrected 400-mutation result; the earlier
`post-fused4-screen-v2.json` was a smaller successful screen.

Archive `detailed-marker-evidence.tar.gz`: 171 files, 3,493,132 bytes.
SHA256 `47395421807aea53697be6090a18348a98605de41770179861934ce162cdbf8e`.
No model weights or large tensor dumps are packaged. All owned services and
component processes exited; AMD-SMI confirmed all eight GCDs free.

The immediate service candidate is exact post reuse. Next, revisit pre-mix
execution without lowering Fn precision, and C4 multi-query K reuse if it
beats the **full** logits/metadata chain. Query-producer compaction and Top-K
alone have much smaller measured budgets here than prior speculation implied.
