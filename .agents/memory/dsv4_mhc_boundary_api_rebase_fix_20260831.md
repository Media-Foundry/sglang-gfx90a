# DeepSeek V4 MHC boundary API rebase fix

Date: 2026-08-31

## Scope

- Original DeepSeek-V4-Flash checkpoint.
- Physical GCDs `HIP_VISIBLE_DEVICES=4,5,6,7`.
- TP4 / EP1 / no A2A.
- Both native AR and DSpark gamma-1 were checked because the repaired helper is
  shared; future performance selectors remain target-verify-only.

## Bug

After the AMD forward-path rebase, the attention boundary still passed
`global_batch_size`, `fn_bf16`, and `fn_fp16` to
`apply_mhc_post_pre_boundary`, but the consolidated helper accepted only the
required keyword `fn_transpose`.  The attention call also omitted that required
keyword.  A fresh import therefore exposed a `TypeError`; an old service or an
inherited package path could hide the mismatch.

The helper now has a backward-compatible `fn_transpose=False` default, accepts
the gfx90a shape/weight metadata, and forwards it to the existing
`mhc_fused_post_pre` decomposition.  Both FFN call sites now pass the same
rank-invariant `ForwardBatch.batch_size` and cached BF16/FP16 mixing weights as
the existing unfused fallback.  This restores the intended dispatch contract;
it does not alter weights, Sinkhorn iteration count, or FP4 math.

## Verification

CPU unit test:

```text
test_deepseek_v4_amd_fused_mhc.py: 10 passed, 4 skipped
```

Fresh DSpark gamma-1 service, 32 concrete coding requests, 256 tokens, four
rounds:

```text
resident tok/s:  760.989 / 781.696 / 783.610 / 788.709
scheduler tok/s: 763.867 / 773.838 / 772.262 / 765.253
host step ms:     70.666 / 71.153 / 71.079 / 69.634
median resident:  782.653 tok/s
median scheduler: 768.758 tok/s
median host step: 70.873 ms
mean accept:      1.7582
```

France first-nine was exact and semantic Paris passed.

Fresh native AR negative-control, 32 concrete coding requests, 256 tokens,
three rounds:

```text
resident tok/s:  721.610 / 729.557 / 732.213
scheduler tok/s: 727.477 / 735.520 / 738.342
host step ms:     43.988 / 43.507 / 43.340
```

France first-nine was exact and semantic Paris passed.  This is not an AR
regression relative to the established roughly 707--717 scheduler tok/s native
range.

Artifacts:

```text
/tmp/dsv4_mhc_fix_france_diverse.json
/tmp/dsv4_mhc_fix_code32_4r.json
/tmp/dsv4_mhc_fix_ar_france.json
/tmp/dsv4_mhc_fix_ar_code32.json
/tmp/dsv4_mhc_api_fix_fresh.log
/tmp/dsv4_mhc_api_fix_ar.log
```

## Guardrail

This commit is an API/correctness repair in shared code.  New performance work
must use an explicit target-verify forward-mode predicate plus exact M64/M128
shape guards so native AR cannot select speculative-only kernels.
