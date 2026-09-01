# DSpark gamma-3 draft-M96 CK sparse attention rejection (2026-09-01)

## Question

The accepted gamma-three service enables the CK/MFMA sparse-decode kernel for
the M128 target-verify tier, but leaves the pre-existing M96 selector disabled.
The DSpark draft transformer processes three draft rows for each of 32 requests,
so its attention naturally reaches M96. Could enabling the existing M96 CK
selector reduce the serial 10--11 ms draft segment?

## Protocol

- Physical GCDs 4--7, TP4 / EP1, original weights.
- Graph tiers only 1 and 32; all other accepted gamma-three profile switches
  unchanged.
- `SGLANG_DSV4_GFX90A_TP4_M96_CK_SPARSE_DECODE=1` was supplied only to this
  DSpark service. No code was modified.
- Fixed heterogeneous workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`.

## Results

The separate BS1 France oracle passed 3/3 exactly and produced the historical
completion hash each time.

BS32, 32 distinct requests, 1024 generated tokens, `stream_interval=1`:

```text
resident: 1562.82 / 1583.84 / 1547.86 tok/s
median:   1562.82 tok/s
accept:   3.519 / 3.554 / 3.488
```

The relevant control services centered around 1565--1572 tok/s. Thus the M96
candidate is neutral to slightly negative end to end even though its standalone
attention core is faster. Three draft layers do not provide enough attention
budget for the kernel gain to survive graph scheduling and service noise.

## Decision

Keep the generic M96 selector available for its earlier gamma-two target
experiments, but do not enable it in the gamma-three TP4 profile. This does not
affect native AR. Further draft-only attention tuning is below the current 2k
budget; target-verify structural work remains the priority.
