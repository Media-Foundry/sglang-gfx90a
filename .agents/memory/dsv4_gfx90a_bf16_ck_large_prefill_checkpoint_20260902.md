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

## Small configuration sweep

The full helper was subsequently swept on physical GPU 4 with balanced and
deliberately skewed routes.  All reported values include both FP4-to-BF16
weight expansions and the complete CK two-stage MoE.

| M | route | blocks=832 | blocks=1664 |
|---:|:---|---:|---:|
| 8192 | balanced | 7.282 ms | 7.227 ms |
| 8192 | skewed | 7.049 ms | 6.983 ms |
| 12288 | balanced | 8.740 ms | 8.696 ms |
| 12288 | skewed | 9.326 ms | 9.259 ms |
| 16128 | balanced | 9.301 ms | 9.264 ms |
| 16128 | skewed | 9.970 ms | 9.943 ms |

Changing 832 to 1664 dequant blocks improves only about 0.3--0.9%, so this
path is not meaningfully limited by that launch geometry.  Further broad CTA
count sweeps are closed.

## ISA/CK audit

The production SDOT contract was rechecked against the CDNA2 ISA: packed FP4
maps exactly to the doubled signed-I8 codebook, the combined per-group scale
must include `0.5`, the E8M0 zero exponent handling is correct, and the wave64
fixed reduction tree is valid.  CK's high-level M32 I8 alias must not be used
on gfx90a because that specialization converts I8 operands to FP32 MFMA.  Only
the low-level native gfx90a I8 primitive is reusable.  The remaining bounded
MFMA experiment is a custom native M16N16K16 schedule intended to reduce
accumulator/VGPR pressure; it must beat the complete production gate by at
least 5% before integration.
