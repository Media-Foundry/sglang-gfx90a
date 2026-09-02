# DSV4 gfx90a BF16-CK large-prefill checkpoint (2026-09-02)

## Scope

This is a default-off experimental path for original DeepSeek-V4-Flash FP4
weights on TP4/EP1/no-A2A.  It expands routed weights into reusable BF16
workspaces and invokes AIter/CK variable-M grouped MoE only for
`8192 <= M <= 16384`.  Decode and the ordinary M2304 C1 prefill path do not
enter this selector.

## Standalone result

The full helper includes both FP4-to-BF16 weight expansions, CK sorting,
gate/up, activation, down projection, and output reduction.

- M4608 balanced routing: 5.851 ms median.
- M4608 deliberately skewed routing: 6.287 ms median.
- Repeated replay was stable.

The comparable production raw-FP4 routed core is roughly 24 ms at M4608, so
this justified an end-to-end experiment despite the extra BF16 workspace.

## End-to-end C32 result

The successful admission profile used a 32768-token pool, a 16384-token
prefill limit, seven queued requests per large step, and a deliberately long
5 s experimental prefill-delay cap.  The scheduler formed M16128 batches and
a final M2303 tail.

- First 32-request run: 73,724 real heterogeneous code-prompt tokens in
  17.2486 s = 4274.20 input tok/s.
- Warm rounds: 4104.81, 4012.98, 4090.61 input tok/s.
- Warm median: 4090.61 input tok/s.
- Observed memory after workspace/KV allocation: about 56.8--58.3 GiB/GCD.

This is about 49% above the accepted ~2.74k C32 baseline and above the older
~3.07k large-M AIter experiment, but remains far below the 10k input tok/s
goal.  The 5 s delay is an oracle configuration, not a production admission
policy and is unacceptable for C1.

## Correctness and isolation

- Standard-path native AR with the experimental selector enabled but not hit:
  52.209 / 52.051 / 52.099 tok/s, identical completion hash
  `38c3d431e7c1dd65`.
- The France semantic test on the ordinary small-M path returned `Paris.`.
- All C32 requests completed, but cross-round first-token hashes were not
  exact.  Therefore large-M numerical correctness is not accepted yet.
- No successful fresh 4604-token C1 measurement was obtained in this round;
  the selector is statically unreachable for M2304 and remains default off.

## Status

Keep this as a default-off oracle/checkpoint.  Do not enable it in the normal
launch profile until teacher-forced/semantic correctness of the actual
large-M path is established and an adaptive admission policy replaces the
5-second batching delay.
