# TP8 DSpark strict M192 target row-prefetch checkpoint (2026-09-10)

## Scope

- DeepSeek-V4-Flash original checkpoint weights.
- TP8 / EP1 / no A2A, 1,048,576-token pool.
- Strict DSpark gamma five, static uniform verification: C32 produces M192.
- No anchor-only routed approximation and no compact target budget.
- 32 fixed heterogeneous code requests, 1024 generated tokens/request.

The selector is limited to target verification width six, C32, exact M192,
TP8/EP1, gfx90a, I256 and the A4/R2/R2 grouped geometry. Native AR, gamma-three
M128, draft M160 and prefill cannot enter it.

## Component oracle

Physical GCD 4, 100 input/weight mutations, 1000 HIP Graph replays, bitwise
exact against the current grouped kernel:

| routing | current | row-prefetch | improvement |
|---|---:|---:|---:|
| balanced | 1095.100 us | 1011.844 us | 8.228% |
| skewed | 1042.549 us | 988.966 us | 5.418% |

Evidence:

- `/tmp/dsv4_tp8_m192_rowprefetch_balanced.log`
- `/tmp/dsv4_tp8_m192_rowprefetch_skewed.log`

## Fresh-process service ABBA

| arm | resident tok/s rounds | median | host step ms |
|---|---|---:|---:|
| A control | 842.858 / 838.032 | 840.445 | 116.126 / 115.509 |
| B candidate | 968.063 / 972.207 | 970.135 | 100.738 / 97.478 |

Median gain: **15.43%**. Mean acceptance stayed in the same range and was
slightly lower for B (`3.319 / 3.329`) than A (`3.373 / 3.325`), so the gain is
from a shorter strict target step rather than acceptance drift.

Correctness and quality gates:

- France C32: 32/32 exact for both fresh A and B services.
- Every request produced 1024 tokens with `finish=length`.
- Across 64 candidate completions, median repeated-8gram ratio was 0.02065,
  p95 0.06883, maximum 0.36971 and zero exceeded 0.8. Control median was
  0.01917 and likewise had zero severe repetitions.
- The component oracle is bitwise exact. Full generated sequences are not
  cross-round bitwise identical because the surrounding strict DSpark path has
  pre-existing low-order/acceptance-order variation; no new semantic collapse
  was observed.

Artifacts:

- `/tmp/dsv4_tp8_gamma5_m192_A2_c32x1024.json`
- `/tmp/dsv4_tp8_gamma5_m192_B_c32x1024.json`
- `/tmp/dsv4_tp8_gamma5_m192_A2_france.json`
- `/tmp/dsv4_tp8_gamma5_m192_B_france.json`

## Decision

Accepted and default-enabled only when the explicit TP8 full-target profile is
run with gamma five. Gamma three production remains unchanged. This moves the
strict gamma-five uniform target from roughly 0.84k to 0.97k resident tok/s;
the 2k objective remains open and requires both higher acceptance and further
target-step reduction.
