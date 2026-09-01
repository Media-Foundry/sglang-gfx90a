# DSpark gamma-3 BS32 segment profile (2026-09-01)

## Scope

- Physical GCDs `4,5,6,7`, TP4 / EP1, original checkpoint weights.
- Accepted static gamma-three profile with M128 target verify and CK sparse
  attention.
- Frozen 32-request heterogeneous workload selected with seed `8675309`.
- Workload SHA256:
  `f74de67a93a660cde060991df71c9e2972a05d82c3ba3f9fe7c144b1f066a152`.
- Observer components: `core`, `step_cpu_time`, `step_gpu_time`,
  `draft_gpu_time`, and `target_verify_gpu_time`; `reqs` was intentionally
  disabled to avoid D2H staging.

The normal `/server_info` control RPC returned 502 while the post-warmup
`freeze_gc` retry loop was active. A temporary log-only diagnostic emitted one
record every 16 steps and was removed immediately after collection; no model
or production observer code remains changed.

## Stable BS32 samples

| sample | CPU step ms | GPU step ms | draft ms | target verify ms | GPU residual ms |
|---:|---:|---:|---:|---:|---:|
| 16 | 69.072 | 69.040 | 9.953 | 56.632 | 2.455 |
| 32 | 71.331 | 71.367 | 11.478 | 57.433 | 2.456 |
| 48 | 71.351 | 70.798 | 10.719 | 57.605 | 2.474 |
| 64 | 71.507 | 71.028 | 11.202 | 57.366 | 2.460 |

The CPU and GPU step intervals nearly coincide, so there is no material host
submission gap to optimize. Draft plus target verify account for roughly
`96.5%` of the GPU step. Target verify alone is about `80--81%`; draft is
about `14--16%`; all planning/acceptance/residual GPU work is only about
`2.46 ms/step`.

## Consequences

- HTTP, scheduler receive, and generic host glue cannot close the 2k gap.
- A 3.55-token average commit needs approximately `57--64 ms/step` for 2k,
  depending on the exact resident accounting. The observed 69--71 ms step
  therefore needs a structural 7--14 ms reduction.
- Complete target/draft graph concurrency is already rejected because both
  graphs contain TP collectives and deadlock their epochs.
- The strongest remaining exact direction is the previously proven
  semantic-lane compute overlap: keep all collectives in one global order,
  while overlapping anchor routed M32 with draft shared-M96 followed by
  draft CK-attention M96 between communication boundaries. Its compute-only
  oracle hid `191.74 us/layer` (about `8.25 ms/43 layers`).

Continuation gate for a four-rank prototype:

1. no concurrent or reordered collective;
2. at least `100 us/layer` or `4.3 ms/step` net E2E reduction;
3. separate BS1 France exact;
4. frozen heterogeneous workload SHA unchanged;
5. acceptance length must not fall by more than `0.03`;
6. resident throughput gain at least 5%, otherwise remove the prototype.
