# DSV4 TP4 DSpark concurrency and reference audit (2026-08-31)

## Scope

- Hardware: four MI250 GCDs, alternating to `HIP_VISIBLE_DEVICES=4,5,6,7`
- Target: original DeepSeek-V4-Flash checkpoint, TP4/EP1/no-A2A
- Workload: 32 distinct, concrete coding prompts from
  `dsv4_tp4_code_32_input_ids.json`
- Correctness oracle: official France prompt, greedy completion IDs and semantic
  Paris check

## Apparent lack of concurrency

The service can run 32 requests concurrently.  Stream timelines showed a
9--19 s common resident window and scheduler logs reached BS32.  Two admission
effects created the misleading serial appearance:

1. With a 256-token page and `chunked_prefill_size=8192`, a 32-request short
   prompt batch sat on the exact 32-page boundary and was observed as 31+1.
2. The experimental PrefillDelayer used `queue_min_ratio=1.0` and a 5000 ms
   timeout.  Once 31 requests were running and one waited, the one-request queue
   could never meet the derived minimum of 31, so it waited for the timeout.

Disabling the delayer removed the five-second tail.  Raising the prefill budget
to 8448 removed the capacity boundary, but independent HTTP arrivals still
formed 1+31.  `batch_notify_size=32` did not coalesce intake and is not an input
batching control.

## Verify graph cliff

The corrected SPS profiler and forced-budget bypass measured a real BS32 cliff:

| Verify graph | Step time |
|---|---:|
| M96 | 177--181 ms |
| M128 | 304--305 ms |

The additive SPS table linearly interpolates between sparse M probes and cannot
represent this graph-tier discontinuity.  Real compact steps at 245--257 ms are
consistent with a mixture of M96 and M128/M192, not a large CPU-only gap.  The
calibration used for these diagnostics is stored in
`.agents/experiments/dspark_sps_tp4_fixed_20260831.json`.

## Correctness isolation

- Native AR on GCDs 4--7 produced the exact France prefix
  `[671, 6102, 294, 8760, 344, 2619, 51119, 42499, 1]` and semantic Paris.
- Gamma-5 compact DSpark diverged by token 3 and was not semantically correct,
  even with forced full verify budget.  SPS pruning is therefore not the root
  correctness problem.
- Gamma-1 static reference DSpark, with folded proposal/sampling, stacked KV,
  embedded lookup, multistream and overlap disabled, stayed correct through
  token 4 and answered Paris, but was not token-exact and repeated Paris/EOS.
- Disabling CUDA Graph produced the same gamma-1 trajectory, so capture itself
  is not the first-divergence cause.

## Eager DSpark embedding bug and fix

Before the fix, the no-graph/no-embed reference path crashed in the first draft
MHC block:

```text
mat1 and mat2 shapes cannot be multiplied (1x4096 and 16384x24)
```

The eager path called only the shared vocabulary embedding and flattened it to
`[rows, H]`; DSV4 DSpark requires model-specific `forward_embed`, which expands
to `[rows, hc_mult, H]`.  `build_eager_draft_input_embeds` now reuses the draft
model hook when present and preserves the historical flat lookup for generic
drafts.  GPU validation showed that the reference service no longer crashes.

Seven focused unit tests passed.  The remaining non-exact trajectory is now
isolated below embedding preparation, SPS, folded proposal, multistream,
overlap, and CUDA Graph; next work should compare multi-token target-verify
cache/position progression against sequential native decode at the first
France divergence.

## Performance notes (not checkpoints)

- Compact gamma-5 BS32, delayer disabled, interval-1 diagnostic: resident
  156.17 tok/s.  It failed correctness and is not an accepted result.
- Native AR BS32 on GCDs 4--7: aggregate median 432.74 tok/s, scheduler decode
  about 450--457 tok/s, about 70 ms/step.  This is below earlier native results
  and needs a configuration/worktree regression audit before reuse as baseline.
- Eager gamma-1 DSpark was only 5.78 tok/s; it is a correctness oracle, not a
  performance configuration.

## Restored native target baseline

An old diagnostic environment had carried
`SGLANG_DSV4_GFX90A_BF16_ATTN_LINEAR=0` into later services.  This was the
concrete cause of the apparent native M32 regression:

| Target projection profile | Scheduler decode | Host step |
|---|---:|---:|
| BF16 attention linear disabled | about 453 tok/s | about 70 ms |
| BF16 attention linear enabled | 721--740 tok/s | 43.25--44.38 ms |

With the shipped default restored, France was exact and semantic Paris.  Three
256-token, 32-code-request rounds produced common-resident throughput of
633.30, 643.85, and 649.47 tok/s.  Aggregate wall throughput remained unstable
because intake/stream seams added long non-resident intervals in two rounds;
scheduler and common-resident numbers are the reliable target-side baseline.

## Markov W2 rejection

The original FP32 replicated Markov W2 was tested against the optimized BF16,
TP-sharded path.  At gamma 5 / BS32 it retained essentially the same poor draft
quality (mean accept length 1.565, draft-position accept rate 11.0%) while
running slower.  France remained semantic Paris.  The W2 dtype/sharding
optimization is therefore not the cause of low acceptance and should remain
enabled.

## M96 graph-tier alignment ABBA

With target BF16 attention restored, forced budget fraction 0.2 selected the
M96 graph at BS32 but scheduled only 64 real verify tokens.  Alignment fills
the already-paid padding slots without selecting a larger graph.

| Order | Alignment | Resident tok/s | Scheduler tok/s | Mean accept |
|---|---|---:|---:|---:|
| A1 | off | 276.66 | 335.58 | 1.425 |
| B1 | on | 321.72 | 372.98 | 1.634 |
| B2 | on | 275.80 | 343.58 | 1.376 |
| A2 | off | 276.32 | unavailable | 1.470 |

The two-point center is 276.49 tok/s for A and 298.76 tok/s for B, an 8.1%
resident improvement.  Step time stayed in the same M96 range (roughly
91--100 ms); France remained semantic Paris.  Because B is noisy, future
checkpoints should retain multiple real-prompt rounds, but the mechanism and
ABBA center justify enabling alignment by default only in the measured
gfx90a TP4 BS32 DSpark profile.  It remains explicitly overridable with zero.
