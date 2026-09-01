# DSpark M128 shared bounded-SwiGLU fusion rejection (2026-09-01)

## Workload protocol

- Physical GCDs: `HIP_VISIBLE_DEVICES=4,5,6,7`
- TP4 / EP1, original checkpoint weights, DSpark gamma 3
- 32 distinct real token-ID requests sampled once with seed `8675309`
- Frozen workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`
- Every service and round reused the same `input_ids`; only `cache_salt`
  changed to prevent prefix-cache reuse.
- `stream_interval=1`, 1024 generated tokens/request.

## Candidate

The gfx90a shared-expert fallback expands bounded SwiGLU into separate Torch
clamp, SiLU, clamp, and multiply kernels. A temporary vectorized HIP kernel
fused this chain for the strict TP4 `TARGET_VERIFY`, `M=128`, gate/up
`[128,1024]` tier. It explicitly rounded the SiLU result to BF16 before the
final multiply, matching the existing Torch operation boundary.

Standalone HIP-graph ABBA:

- Torch chain median / trimmed mean: `17.143 / 17.144 us`
- fused HIP median / trimmed mean: `6.206 / 6.205 us`
- isolated saving: `10.939 us/layer` (`63.8%`)
- 100/100 changing-input mutations bitwise exact
- maximum absolute error and relative L2: zero
- 1000 graph replays bitwise stable

The path was guarded by an explicit DSpark profile switch plus
`TARGET_VERIFY`, exact M128, and exact gate/up shape, so native AR could not
enter it.

## Correctness

The service captured successfully. The separate BS1 France oracle passed
3/3 exactly with identical completion SHA256
`93a6e7ee9273cca3c4eeab558c715185459a144292177bc0884e88c95b275665`.

## Service ABBA

Resident BS32 throughput, tok/s:

| Arm | rounds | median |
|---|---|---:|
| A1 control | 1572.30 / 1598.63 / 1572.51 | 1572.51 |
| B1 fused | 1557.01 / 1609.00 / 1558.95 | 1558.95 |
| B2 fused, independent service | 1560.48 / 1551.17 / 1566.25 | 1560.48 |
| A2 control, independent service | 1507.57 / 1588.58 / 1559.07 | 1559.07 |

The two fused services centered at `1559.71 tok/s`; the two control service
medians centered at `1565.79 tok/s`, approximately `-0.39%` for the candidate.
The low first A2 round shows the remaining service noise, but neither fused
service produced a positive center.

## Decision

Reject and remove the production selector/kernel. The isolated elementwise
gain is real and exact, but the shared expert already overlaps the routed
branch on the alternate stream. Shortening a non-critical branch does not
shorten the layer join; the extra graph node/schedule perturbation is neutral
to slightly negative end to end. Do not retry bounded-SwiGLU alone unless a
future trace shows shared expert becoming the slow join branch.
