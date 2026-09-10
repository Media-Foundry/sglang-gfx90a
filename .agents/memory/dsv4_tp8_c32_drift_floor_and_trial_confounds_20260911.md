# TP8 C32 drift floor and fused-MHC trial confounds (2026-09-11)

## Scope

Strict TP8 DSpark full-target profile, C32, gamma three, original weights, 1M
token pool, real 32-request code manifest
`/tmp/dsv4_open_code_pd_20260908/decode.json` (512-token inputs), 512 generated
tokens, greedy, `ignore_eos`, stable cache-salt namespace. One control arm
completed; the run was stopped before the candidate arms produced measurements.

## The control's own instability is the headline

Arm A1 is the control -- no candidate change, `MHC_FUSION=0`. Two measured
waves against one warm service, warm wave excluded:

| wave | aggregate tok/s | severe repetition |
|---|---:|---|
| 0 | 637.04 | none |
| 1 | 554.75 | request 27 |

In-arm wave0 -> wave1: **32/32 requests diverge**, median first differing token
23 of 512, minimum 0, mean shared prefix fraction **0.0814**.

Three consequences for any future candidate judgement:

1. A differing-request count saturates at 1.0, so `cross <= floor` is
   unfalsifiable. Use divergence-onset metrics that retain resolution at
   saturation: first-differing position and shared-prefix fraction.
2. Severe repetition appears in the control, so it is not a candidate-specific
   quality signal at this baseline.
3. Throughput swings 12.9% between two waves of the same service. The entire
   M128 MHC pool is worth about 2.4% of the step, so two waves per arm cannot
   resolve the effect under test.

The 637/555 tok/s figures are not comparable to the ~1077-1088 tok/s resident
baseline: this harness measures whole-wave wall time including admission and
drain, not a common resident decode interval.

## Three confounds found in the trial itself, not in the candidate

The run was stopped after A1 because the arms were not comparable:

- **Unpinned random seed.** `RANDOM_SEED` unset lets each fresh service choose
  its own: A1 got 172686631, B1 got 881416104. Control and candidate therefore
  differed by seed as well as by candidate. The launcher already supports
  `RANDOM_SEED`; it simply was not set.
- **Accept length never recorded.** `avg_spec_accept_length` is only added to
  `/server_info` when metrics are on (`spec_total_num_forward_ct > 0` guard in
  `Scheduler.get_internal_state`), and `ENABLE_METRICS` defaults to 0. A1
  recorded `accept_lengths=[]`, so a candidate could have bought step time by
  accepting fewer draft tokens undetected.
- **The candidate conflated two numerical changes.** With
  `FP16_MHC_DOT=1` (launcher default) the split-K fused tail consumes `fn_fp16`,
  so admitting it on C32 changes the mixing weights from fp32 to fp16 *and* the
  reduction order. Measured at M128 on real layer-20 tensors:

| effect | resulting `mixes` max_abs | relative |
|---|---:|---:|
| fp16 mixing weights | 3.75e-03 | 140x |
| split-K reduction order | 2.67e-05 | 1x |
| `BLOCK_N` geometry | 3.8e-06 -- 1.1e-05 | 0.4x |

  The weight rounding changes 393173/393216 weights and 3069/3072 outputs, so it
  dominates. A B-arm result could not have said which cause moved the tokens.

## Harness defects found by verifying assumptions against source

Each of these was found by reading the source or querying a live service, and
each would have corrupted a verdict silently:

- `/server_info` spreads resolved server args at the **top level**; reading
  `.get("server_args", {})` returned `{}`, so the arm-config guard passed on
  nothing. `/get_server_info` is a deprecated alias.
- A non-stream finished `/generate` response carries `output_ids` at the top
  level, not under `meta_info`. An empty list read as perfect agreement, i.e. a
  false no-drift verdict.
- `start-dspark` already daemonizes and runs the launcher's own `wait_ready`
  (log marker + TCP connect + `freeze_gc`), whose source comment warns against
  HTTP readiness polling -- exactly what the driver had added. `$!` captured the
  wrapper shell, not the server.
- A shared `LOG_FILE`/`PID_FILE` let `is_running` turn an arm's start into a
  no-op, silently measuring the previous arm's env.

## Next step: establish a reproducible baseline first

`--enable-deterministic-inference` (batch-invariant ops) is **compatible** with
this profile: its only speculative conflict is with
`--speculative-use-rejection-sampling`, which the strict profile leaves False.
It was not enabled (`enable_deterministic_inference: False` in A1) and the
launcher had no pass-through; one was added, default off.

It is a measurement config, not a throughput profile -- batch-invariant ops are
expected to cost speed. Its value is that on a reproducible baseline a
candidate's divergence becomes attributable instead of vanishing into a 1.0
noise floor. Native AR at TP8 is already fully reproducible
(`dsv4_tp8_drift_fix_validation_20260907`, six gates), so the irreproducibility
is specific to the speculative path.

Planned arm order, now six arms:

```text
A1:fusion=0,fp16=1   B1:fusion=1,fp16=1   C1:fusion=1,fp16=0
C2:fusion=1,fp16=0   B2:fusion=1,fp16=1   A2:fusion=0,fp16=1
```

C isolates reduction order from weight rounding. All arms pin
`RANDOM_SEED=20260911`, enable metrics, and enable determinism.

Open question before re-running: whether determinism actually collapses the
in-arm divergence to zero. If it does not, the candidate cannot be judged E2E at
all and the next move is a teacher-forced per-layer comparison instead of a
service A/B. Do not spend the ~47 us/layer pool on a service ABBA until an arm
reproduces itself.

Artifacts: `/tmp/dsv4_tp8_mhc_fusion_drift_20260911/arm_A1.json`,
`server_A1.log`. Diagnostic: `/tmp/mhc_fp16_weight_effect.py`.
