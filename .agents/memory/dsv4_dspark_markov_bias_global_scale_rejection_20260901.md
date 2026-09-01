# DSpark global Markov-bias scale rejection (2026-09-01)

## Scope

- Physical GCDs 4--7, TP4/EP1/no-A2A, original checkpoint weights.
- Gamma-three DSpark, early exact tier 33, steady compact M128 target verify.
- Frozen 32-request heterogeneous workload, 1024 emitted tokens/request,
  `stream_interval=1`.
- The diagnostic scaled only the DeepSeek-V4 DSpark draft Markov correction
  before adding it to the base logits. Target verification and native AR were
  unreachable from the selector.

## Result

| Global bias scale | Resident BS32 tok/s | Mean accepted length | France first-nine / semantic |
|---:|---:|---:|---|
| 0.75 | 1515.87 | 3.4259 | pass / pass |
| 1.25 | 1501.10 | 3.4099 | pass / pass |

The accepted scale-1 checkpoint is about 1501 tok/s with mean accepted length
about 3.42 on the same class of run. Neither direction improved acceptance;
the 0.75 throughput difference is ordinary single-round service variance and
does not meet the acceptance gate of 3.59 or the performance gate of +5%.

Artifacts:

- `/tmp/dsv4_markov_lambda075_bs32_1024.json`
- `/tmp/dsv4_markov_lambda125_bs32_1024.json`

## Decision

Reject a single global Markov correction scale and remove the experimental
runtime knob. A future per-draft-position calibration is a distinct experiment
and must use held-out request traces; do not infer it from this global screen.

## Current steady marker decomposition

A diagnostic layer-20 realtime-marker service (marker overhead excluded from
performance claims) observed the compact steady target layer at roughly
760 us rank-max:

- attention prepare: 240--246 us;
- attention output: 88--90 us;
- attention-entry MHC/Norm: 82--85 us;
- FFN-entry MHC/Norm: 128--138 us;
- compact MoE: about 175 us.

The early exact tier still spends about 565--580 us in MoE. This confirms that
the steady critical path has moved away from routed MoE toward attention
preparation plus the two MHC/Norm boundaries.
