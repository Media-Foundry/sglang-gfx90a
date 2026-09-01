# DSV4 prefill AIter block-M sweep (2026-09-02)

## Setup

TP4/EP1, no A2A, native checkpoint, preshuffled AIter/CK FP4 path,
`chunked_prefill_size=16384`, physical GCDs 4--7. Each round used the same
manifest of 32 distinct real code-review requests (73724 audited prompt tokens),
but each request had its own cache salt. Throughput is measured from first HTTP
send to the last first streamed token.

An experimental fail-loud environment selector was temporarily wired directly to
`aiter.fused_moe.cfg_2stages[key]["block_m"]`. Runtime insertion was exercised for
each candidate; the selector was removed after the negative result.

## Results

| block_m | rounds (input tok/s) | warm center | cross-round first-token exact |
|---:|---|---:|:---:|
| 32 | 2930.62 / 3095.72 / 3127.88 | 3111.80 | yes |
| 64 | 3003.66 / 3132.79 / 3131.46 | 3132.13 | yes |
| 128 | 2914.16 / 3169.80 / 3092.14 | 3130.97 | no |

Artifacts:

- `/tmp/dsv4_prefill_c32_aiter_ksplit0_abba.json`
- `/tmp/dsv4_prefill_c32_aiter_blockm64.json`
- `/tmp/dsv4_prefill_c32_aiter_blockm128.json`

## Decision

Keep block-M 32. The 64/128 warm centers are only about 0.6% above the control,
well inside run-to-run variation, and 128 loses the cross-round exactness witness.
Changing sorter padding granularity alone does not reduce the dominant grouped
expert weight traffic. Do not repeat this metadata sweep unless the grouped kernel
work decomposition changes.
