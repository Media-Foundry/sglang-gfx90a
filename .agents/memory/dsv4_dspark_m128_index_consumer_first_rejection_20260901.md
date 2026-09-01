# DSpark M128 index-consumer-first rejection (2026-09-01)

The strict TP4/C4/M128/BS32 `TARGET_VERIFY` candidate delayed the independent
core-compressor stream join until after the indexer consumer. Native AR was
unreachable. Dependency review confirmed separate core/index caches and no
missing mathematical edge, but concurrent kernels compete for CU/L2/HBM.

Real heterogeneous 32x1024 results, original weights and stream interval 1:

```text
candidate: 1450.76 / 1479.49 / 1512.60, median 1479.49 tok/s
rollback:  1509.03 / 1527.68 / 1440.17, median 1509.03 tok/s
```

All six rounds passed the France first-nine and semantic Paris checks. Median
resident throughput divided by mean accepted length was about 434 for the
candidate and 435 for rollback, so the scheduling change is neutral after
acceptance normalization and negative in raw median. This matches an older
TP8/M32 non-reproducible result. Remove the selector; do not retry without a
resource-partitioned kernel.

Artifacts:

- `/tmp/dsv4_index_consumer_candidate_bs32_1024_r3.json`
- `/tmp/dsv4_index_consumer_rollback_bs32_1024_r3.json`
