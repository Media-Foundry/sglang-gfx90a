# DSpark gamma-3 M128 1-MiB AR and routed Top-4 screens (2026-09-01)

## TP4 1-MiB AIter all-reduce geometry

Gamma-three target verification reduces BF16 `[128,4096]`, exactly 1 MiB,
after both attention and MoE.  The existing AIter patch only exposed the
512-KiB M64 geometry.  A second, default-off hook now matches only
`gfx90a + world_size=4 + bytes=1MiB` through
`AITER_GFX90A_AR_1M_BLOCKS`; the layer-20 registered-buffer oracle accepts
`--tokens 128`.

Initial graph screen (`blocks=8/12/16/24/32/48/64/80`) found AR gross times:

```text
8    53.219 us
12   45.903 us
16   46.530 us
24   47.955 us
32   51.234 us
48   54.049 us
64   61.507 us
80   62.980 us
```

Full 100-mutation/1000-replay/7-round validation remained bitwise exact:

```text
blocks=12  44.815 us gross
blocks=16  45.843 us gross
blocks=80  62.357 us gross
```

The isolated saving is real (about 17.5 us, 28%).  Service results were not
stable enough to promote:

- blocks=12: first 256-token round reached 1188.16 resident tok/s, but France
  failed in round 1;
- blocks=16: five 64-token rounds passed, then the next 256-token France gate
  failed immediately;
- blocks=24: four accepted rounds were
  `1159.95/1156.28/1214.15/1209.83 tok/s` (center about 1184.89), but France
  failed in round 4;
- blocks=80 control with the experimental two-stage final barrier also failed
  France in round 1, so the barrier did not cure the broader approximate-path
  drift.

The suspected peer-temp lifetime fix added a final rank rendezvous after the
two-stage all-gather.  It increased blocks=12 AR from about 44.8 to 47.5 us and
did not stop service drift; it was removed.  The production AIter module was
restored to the pre-final-barrier binary.  The 1-MiB geometry hook remains
default-off for future race localization, not as a profile default.

Artifacts include `/tmp/dsv4_ar1m_*` and
`/tmp/dsv4_ar1m_finalbarrier_*`.

## Learned-router Top-4 mass pruning

A strict speculative-only quality screen kept hash layers 0--2 at Top-6 and,
for learned layers, retained the four largest anchor-route weights while
preserving the original total routed mass.  Guards required the existing
gfx90a TARGET_VERIFY/BS32/width-four/M128 pre-router compact path; AR could not
enter.

France failed on the first round even after correcting mass preservation:

```text
671 13102 20702 123327 20702 82318 18099 63376 20702
semantic Paris: false
```

The temporary selector and tensor-op spelling were removed.  Do not proceed
to a fused HIP Top-4 kernel: the quality gate rejects the approximation before
kernel overhead matters.

## Decision

- Keep Top-6 routed experts.
- Keep the 1-MiB AR hook and M128 oracle only as diagnostic infrastructure.
- Do not default to fewer than 80 blocks until the cross-round DSpark
  correctness drift is localized and a long real-various run passes.
- Continue with exact M128 attention scheduling/producer-consumer work.

### Subsequent resolution

The 80-block decision above is superseded for the TP4/BS32 DSpark profile by
the later composed ABBA.  Twelve-block 1-MiB AR plus exact M32 gate-row
prefetch passed three 32x1024 diverse-request candidate rounds at
1531.62/1547.83/1544.34 tok/s with France 3/3, while the full rollback reached
1439.50/1472.90/1378.69 tok/s.  A fresh launcher-default service also passed
France 3/3 and centered above 1500 tok/s.  See
`dsv4_dspark_ar12_m32_rowprefetch_checkpoint_20260901.md`; the older warning
still applies to 12-block AR used alone or outside that strict profile.
