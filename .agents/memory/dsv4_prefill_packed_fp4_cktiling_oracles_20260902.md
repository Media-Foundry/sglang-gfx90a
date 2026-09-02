# DSV4 prefill packed-FP4 CK tiling oracles (2026-09-02)

## Scope and invariant

- Physical gfx90a GCD 4 only for standalone GPU measurements.
- Original DeepSeek-V4-Flash FP4 routed-expert weights.
- Production TP4 prefill selector was not changed.
- `amd-smi process --general --sort-by-pid` was checked before every GPU run.

## Direct M1 access to AIter-preshuffled weights

The current raw-weight wave64 M1 gate/down kernels were temporarily given an
oracle-only addressing mode for AIter-preshuffled weights.  One hundred input,
route-weight and route-ID mutations were bitwise exact, but the layout lost:

| full routed M1 oracle | median latency |
|---|---:|
| raw checkpoint layout | 52.69 us |
| AIter-preshuffled layout | 69.76 us |

The preshuffled form was about 32.4% slower.  A second 16-row/wave mapping was
also correct (maximum difference about `4.7e-10`) but took 118.39 us.  Sweeping
2/4/8/16 output rows retained the existing two-row raw kernel as the winner.
All temporary production-kernel addressing hooks were removed.

Generic online shuffling is not viable at layer granularity.  For one TP4
layer it cost approximately:

| tensor | shuffle latency |
|---|---:|
| W13 | 3572.36 us |
| W13 scale | 295.68 us |
| W2 | 1081.51 us |
| W2 scale | 125.55 us |
| total | 5056.12 us/layer |

## Raw MFMA64 LDS coalescing

A per-wave 256-byte LDS publication/readback was tested around the raw FP4
MFMA64 loads.  Outputs remained exact, but the complete M2304 routed stage
regressed from 24.607 ms to 29.459 ms (about 19.7%).  The LDS path was removed.
Do not retry this per-K-group publication scheme; it adds a barrier and LDS
round trip without providing cross-wave weight reuse.

## AIter CKTile block-M and N-tile sweep

Standalone M4608 A16W4 two-stage CKTile measurements used real DSV4 shapes,
unique Top-6 routes and five-round timing:

| block-M/config | trimmed latency | replay |
|---|---:|---|
| 32, stock N256 | 23.039 ms | not always bitwise |
| 16, stock N128 | 22.427 ms | bitwise exact |
| 64, stock N256 | 25.836 ms | slower |
| 16, experimental N512/4-per-CU | 61.747 ms | not bitwise exact |

Block-M16 is only about 2.7% faster than block-M32, below the 5% component
threshold.  Expanding the N tile to 512 is a decisive rejection: it is about
2.75x slower than the stock M16 candidate, consistent with accumulator/LDS
pressure destroying occupancy on CDNA2.

## Fresh-build dependency issue

A clean gfx90a build exposed two AIter generator defects that stale cached
modules had hidden:

1. FP4 concrete instances were skipped unless `gfx950` was in the target list,
   although the dispatcher selects an `a16w4` table on gfx90a.
2. The main translation unit's explicit implementation includes can become
   stale relative to the generated instance and dispatcher tables.

For the oracle the generator was temporarily allowed to emit FP4 instances on
gfx90a and the explicit includes were synchronized.  All N512 production
configuration changes and the experimental shared object were then removed;
the validated baseline AIter module was restored.

## Decision

Do not connect any of these candidates to service.  Continue prefill work from
the accepted raw-MFMA64 M2304/M2300 path.  The next structural candidate remains
gate/SwiGLU epilogue quantization only if it preserves CTA ownership and avoids
the already-rejected cross-CTA publication/barrier pattern.
