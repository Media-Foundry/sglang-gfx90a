# DSV4 gfx90a C16 prefill tuning and delayer starvation fix (2026-09-05)

## Contract

- Four physical gfx90a GCDs 4--7, TP4/EP1/no-A2A.
- Original DeepSeek-V4-Flash checkpoint weights.
- 32 distinct code prompts, 73,724 audited input tokens, one generated token.
- Native AR only. No decode arithmetic or graph selector was changed.
- Every changed service arm passed the exact nine-token France oracle twice.

## Baselines

The fixed workload is admitted as two exactly full large-prefill batches under
request-16: M=36,864 and M=36,860. Request-12 produces M=27,648, M=27,646,
and M=18,430.

| profile | five-round median | trimmed mean | decision |
|---|---:|---:|---|
| request-12 accepted baseline | 5,744.65 tok/s | not retained in old record | production profile |
| request-16, 36,864-token ceiling | 5,594.88 tok/s | 5,695.75 tok/s | control |
| request-16, 32,768-token ceiling | 5,568.34 tok/s | 5,558.38 tok/s | reject |

The 32,768-token ceiling adds a third large forward and does not improve the
request-16 configuration.

## Component screens

At M=36,864, the established BF16 CK routed path measured 25.6 ms balanced and
23.8--24.1 ms skewed. Stage block-M values 32, 64, and 128 were effectively
tied, so this is not a useful request-16 selector.

hipBLASLt solution 5870 reduced the large-M MHC wrapper from about 5.11 ms to
3.22 ms (1.58x) with cosine 1.0 and finite outputs. Full service, however,
reached only a 5,626.29 tok/s median and 5,600.82 tok/s trimmed mean. It is
therefore rejected as an E2E production tactic.

An FP4-to-BF16 expansion grid of 416 blocks was also rejected: C16 median was
5,543.89 tok/s versus 5,594.88 for control.

## PrefillDelayer starvation bug

With request-16, queue ratio 1, a 5 ms wall bound, and 10,000 pass bound, the
unmodified delayer accepted all HTTP requests but produced no first token for
over one minute. The root cause was that `max_delay_ms` disabled only the
queue trigger. A stale/high recent prefill batch watermark could keep the
independent slot condition true until `max_delay_passes`, turning a 5 ms
batching window into starvation.

The fix applies the wall-clock limit to the combined queue-or-slot delay. A
new unit regression constructs a persistent slot condition and verifies that
it is released as `wait_timeout` after 6 ms despite a 10,000-pass allowance.
The focused pure-function regression and `py_compile` pass. Full pytest
collection in the DS environment is blocked by its unrelated missing optional
`datasets` dependency.

After the fix, the same delayer profile completed normally:

```text
4717.16 cold / 5721.09 / 6095.88 / 5246.83 / 5882.05 tok/s
median 5721.09, trimmed mean 5616.66
```

This proves the starvation repair but, at a 5 ms delay bound, does not show a
stable throughput win over the request-12 profile. The follow-up 20 ms result
below supersedes this intermediate conclusion; the overall profile remains
explicitly opt-in.

## 20 ms aggregation and A-B-B-A validation

Increasing only the prefill wall-clock aggregation bound from 5 ms to 20 ms
made request-16 reproducible. Two consecutive seven-round groups on one B
service were:

```text
B1: 5588.83 / 5889.12 / 5613.74 / 6103.45 / 5744.00 / 5856.53 / 5750.39
    median 5750.39, trimmed mean 5770.76
B2: 5830.73 / 6008.62 / 6204.37 / 5965.77 / 5997.74 / 5772.21 / 6108.39
    median 5997.74, trimmed mean 5982.25
```

The combined fourteen-round B median is 5872.83 tok/s and its one-end trimmed
mean is 5886.73 tok/s. The same-HEAD A2 rollback to request-12 with no delayer
measured a 5659.93 tok/s median and 5645.92 tok/s trimmed mean. Together with
the earlier A1 request-12 median of 5744.65, the symmetric A1-B1-B2-A2 median
centers are approximately 5702 versus 5874 tok/s: request-16 is about 3.0%
faster.

Both services passed the exact France oracle twice. The performance suite
completed all 32 requests every round; its first-token hashes retain the
previously documented non-bitwise large-M BF16-CK variation.

The explicit prefill-throughput profile now defaults to request-16 plus the
20 ms delayer. These settings affect prefill admission only. The native decode
kernel and CUDA-graph selection paths are unchanged.

## Final profile-default acceptance

The profile was relaunched using only
`SGLANG_DSV4_GFX90A_PREFILL_THROUGHPUT_PROFILE=1`, with no manual admission or
delayer overrides. Startup confirmed request-16, M36,864, a 20 ms wall-clock
delay bound, and the unchanged native-AR decode graph tiers 1/2/4/8.

The exact France oracle passed twice. The final seven-round, 32-request diverse
code workload measured:

```text
5405.51 / 6216.98 / 5778.97 / 5806.88 / 5972.60 / 5840.91 / 6222.61 tok/s
median 5840.91, trimmed mean 5923.27
```

This final default-profile median is 1.68% above the earlier request-12
five-round median of 5744.65 tok/s. The stronger symmetric A-B-B-A comparison
above remains the preferred estimate of the scheduling gain because it controls
for service-order drift. Cross-round first-token hashes remain non-bitwise under
the documented large-M BF16-CK path; all requests completed and the independent
France semantic/token oracle stayed exact.
