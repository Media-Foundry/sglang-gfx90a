# DSV4 prefill AIter KSPLIT sweep (2026-09-02)

## Scope

- Hardware: physical GCDs 4,5,6,7 (`HIP_VISIBLE_DEVICES=4,5,6,7`)
- Model: native DeepSeek-V4-Flash checkpoint, TP4/EP1, no A2A
- Prefill budget: 16384 tokens
- Routed path: preshuffled AIter/CK (`SGLANG_DSV4_GFX90A_FP4_DIRECT_MOE=0`)
- Workload: 32 distinct real code-review prompts from
  `dsv4_prefill_diverse_32_input_ids.json`, 73724 audited prompt tokens/round
- Measurement: first HTTP send to last first streamed token, 8 completion tokens

## Results

| KSPLIT | rounds (input tok/s) | warm center | cross-round first-token exact |
|---:|---|---:|:---:|
| 0 | 2930.62 / 3095.72 / 3127.88 | 3111.80 (warm mean) | yes |
| 2 | 2698.35 / 3107.31 | 3107.31 | yes |
| 4 | 2937.91 / 3222.14 / 3113.88 | 3168.01 (warm mean) | no |

Artifacts:

- `/tmp/dsv4_prefill_c32_aiter_ksplit0_abba.json`
- `/tmp/dsv4_prefill_c32_aiter_ksplit2.json`
- `/tmp/dsv4_prefill_c32_aiter_ksplit4.json`

## Decision

Keep `KSPLIT=0` as the accepted configuration. KSPLIT=2 is neutral. KSPLIT=4
shows only about 1.8% higher warm center with overlapping run-to-run variation and
loses the cross-round first-token exactness witness. This is below the threshold
for a production checkpoint and does not justify KSPLIT=8.

The next high-M sweep should target AIter grouped-kernel geometry / `block_size_M`
with proof that the selected value reaches the actual fused-MoE call. Do not retry
generic split-K unless a kernel-level profile shows the reduction cost has changed.
