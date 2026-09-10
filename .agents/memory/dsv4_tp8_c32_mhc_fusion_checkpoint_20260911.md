# TP8 C32 fused-MHC service checkpoint (2026-09-11)

## Scope

Strict TP8 DSpark full-target profile, C32, gamma three, original weights, 1M
token pool, `dsv4` attention backend, accept/draft TP synchronization left on.
Real 32-request code manifest `/tmp/dsv4_open_code_pd_20260908/decode.json`
(16382 input tokens total), 512 generated tokens, greedy, `ignore_eos`, stable
cache-salt namespace, `stream_interval=1`. Each arm is a fresh service with
`RANDOM_SEED=20260911` pinned and its env verified through
`/proc/<pid>/environ`; one excluded warm wave, then two measured waves.

Candidate: admit the split-K fused MHC tail and split-K pre-mix on the C32
target-verify boundary, which passes `global_batch_size=32` and therefore misses
their single-request gates. Scoped to `dspark_m128_active()` -- target-verify
mode, batch 32, verify width 4 -- so native AR, prefill and the draft graph
cannot reach it.

## Measure the resident window, not the wave

Whole-wave wall time is unusable here: the running batch ramps
`0 1 5 10 14 19 23 28` over 8 chunked prefill batches (16382 tokens at
`chunked_prefill_size=2304`), so admission and drain dominate. Same waves, two
metrics:

| arm | resident tok/s | whole-wave tok/s |
|---|---|---|
| A1 control | 1058.43, 1077.73 | 321.66, 563.97 |

Resident spread is 1.8% against 75% for whole-wave, and the resident figure
matches the ~1077 tok/s of the existing baseline. Windows were 12.5--13.4 s and
14165--14868 tokens with all 32 requests contributing.

`--enable-deterministic-inference` is **not available** on this path:
`ValueError: Currently only ['ascend','fa3','fa4','flashinfer','intel_xpu',
'triton'] attention backends are supported for deterministic inference, but you
explicitly specified 'dsv4'`. Batch invariance is therefore not a usable route
to a reproducible C32 baseline, and the earlier plan to get one that way is
closed.

## Result

Arms are A (control), B (candidate, fp16 mixing weights -- the launcher
default), C (candidate, fp32 mixing weights). C exists because with
`FP16_MHC_DOT=1` the split-K tail also swaps fp32 mixing weights for fp16, a
change 140x larger than the reduction-order effect under test (resulting `mixes`
max_abs 3.75e-03 vs 2.67e-05); without C a B result cannot say which cause moved
the tokens.

| arm | config | resident tok/s | mean accept |
|---|---|---|---|
| A1 | control | 1058.43, 1077.73 | 2.674, 2.693 |
| B1 | fusion + fp16 | 1131.23, 1132.45 | 2.684, 2.694 |
| C1 | fusion + fp32 | 1118.05, 1132.14 | 2.691, 2.769 |
| C2 | fusion + fp32 | 1121.76, 1137.96 | 2.704, 2.766 |

Per family, over all measured waves:

| family | n waves | mean tok/s | range | spread | mean accept |
|---|---:|---:|---|---:|---:|
| A control | 4 | 1073.16 | 1058--1082 | 2.2% | 2.687 |
| B fp16 | 4 | 1142.26 | 1131--1160 | 2.5% | 2.698 |
| C fp32 | 4 | 1127.48 | 1118--1138 | 1.8% | 2.732 |

**B +6.44%, C +5.06%** over control. Both exceed every family's own spread. Accepted length is flat to slightly higher
in every candidate arm, so neither gain is bought by accepting fewer draft
tokens.

B and C differ by 1.31%, which is **smaller than B's own 2.5% within-family
spread**, and their ranges overlap at 1131--1138. They are therefore not cleanly
separable on throughput at four waves each; only their separation from control is
established. C recovers most of B's gain while
carrying a 140x smaller numerical perturbation, so C is the configuration to
prefer; a claim that the fp16 rounding is worth a specific fraction of a point
is not supported by four waves per family.

## The gain exceeds the component prediction, unexplained

Component oracles at M128 sized both MHC boundaries at 53.96 us/layer. Against
this trial's implied 80.29 ms step (32 x 2.68 accepted / 1068.08 tok/s) that is
2.89% of the step, yet the measured gain is roughly double it. The single-GCD
oracles timed kernels in isolation on an idle GPU, and a saving on the TP8
critical path where eight ranks synchronize at collectives can be worth more than
its isolated duration -- but that is a hypothesis, not a measurement. Do not
quote 2.89% and ~6% as if they were reconciled.

The older phase profile's 95.825 ms step does not apply to this workload; 80.29
ms is what this manifest and accept length imply.

## Token drift: real, but not attributable

| comparison | differing | first diff median | shared prefix |
|---|---:|---:|---:|
| A1 in-arm (same service) | 30/32 | 7.5 | 0.0992 |
| B1 in-arm | 30/32 | 13.5 | 0.1198 |
| C1 in-arm | 27/32 | 13 | 0.1995 |
| C2 in-arm | 30/32 | 16.5 | 0.1204 |
| A1 vs B1 | 32/32 | 8.0 | 0.0344 |
| A1 vs C1 | 32/32 | 3.0 | 0.0171 |
| A1 vs C2 | 32/32 | 6.5 | 0.0239 |

Cross-config divergence is measurably larger than the in-arm floor (shared
prefix 0.017--0.034 against 0.099--0.200), which is the expected sign for a
non-bitwise change. But the control diverges from *itself* on 30/32 requests, so
this cannot isolate the candidate: the floor is too high for a differing-count or
a prefix fraction to carry an attribution.

The cause is scheduling, not kernel numerics. The batch ramps
`0 1 5 10 14 19 23 28` while chunked prefill admits requests, and MoE routing and
attention geometry are batch-dependent, so an already-decoding request sees a
different batch shape depending on admission timing. That is why deterministic
inference was the wrong instrument even before it turned out to be unavailable:
native AR is already reproducible on this same `dsv4` backend
(`dsv4_tp8_drift_fix_validation_20260907`, six gates).

## Answer quality: the repetition gate was measuring the harness

Both arms produce coherent, on-task answers; request 3's text is near-identical
across arms and request 0 differs in wording while both correctly audit the
source they were given.

Classifying the tails separates two different things:

| arm | dot-pad (ignore_eos filler) | semantic repetition |
|---|---|---|
| A1 control | 3, 1 | 0, 0 |
| B1 fp16 | 1, 0 | **1 (req 5)**, 0 |
| C1 fp32 | 2, 0 | 0, 0 |
| C2 fp32 | 1, 0 | 0, 0 |

Most flags are `ignore_eos` padding a finished answer with `"........."`, and the
**control has more of them than either C arm**. That is a harness artifact, not a
candidate defect; an earlier reading of the raw flag count as a quality signal was
wrong in both directions.

One genuine semantic degeneration occurred in 8 candidate waves: B1 request 5
collapsed into `"1p extension.1p extension..."`. Both C arms and both control
waves are clean. One event is not a reproduced defect, but it is a second reason
to prefer C over B rather than a reason to reject B outright.

## Decision

Accept the fp32 variant (C) and enable it profile-locally in the strict TP8
full-target profile, matching how the draft M96 row-prefetch checkpoint was
wired. Two reasons to take C over B despite B's nominally higher mean: the two
are not separable at this sample size, and B carries both the 140x larger
numerical perturbation and the only semantic degeneration observed in the trial.

The candidate stays behind `SGLANG_DSV4_GFX90A_DSPARK_M128_MHC_FUSION`, scoped to
`dspark_m128_active()`. Native AR, prefill and the draft graph are structurally
excluded: the scope requires target-verify mode at batch 32 with verify width 4,
verified by a truth table (`gbs=1 -> True`; `gbs=32, env off -> False`;
`gbs=32, env on, no scope -> False`; `gbs=32, env on, in scope -> True`).

What this result does **not** establish:

- No drift attribution. The analyzer returns `inconclusive_control_unstable`
  because control diverges from itself on 95% of requests. Cross-config shared
  prefix (0.021--0.026) is lower than the in-arm floor (0.112--0.152), the
  expected sign for a non-bitwise change, but the floor is far too high to
  attribute it. A drift verdict on this candidate is unavailable at this
  baseline, not merely unproven.
- No reconciliation with the component oracles. They sized both boundaries at
  53.96 us/layer, 2.89% of the implied 80.29 ms step, against a measured ~5--6%.
  A saving on the TP8 critical path may be worth more than its isolated
  single-GCD duration, but that is untested.

Artifacts: `/tmp/dsv4_tp8_mhc_fusion_trial_v2_20260911/` (six `arm_*.json`,
`verdict.json`, per-arm server logs).
Reproduce: `scripts/rocm/run_dsv4_mhc_fusion_drift_trial.sh` with
`ARMS="A1:0:1 B1:1:1 C1:1:0 C2:1:0 B2:1:1 A2:0:1"`.
