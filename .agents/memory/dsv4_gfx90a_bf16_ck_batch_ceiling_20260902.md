# DSV4 gfx90a BF16-CK batch-ceiling sweep (2026-09-02)

## Workload and invariant

- Four physical gfx90a GCDs, TP4/EP1/no-A2A.
- Original DeepSeek-V4-Flash packed FP4 checkpoint weights.
- 32 distinct real code prompts from
  `dsv4_prefill_diverse_32_input_ids.json`, 73,724 audited prompt tokens.
- BF16-CK selector remains default off and is unreachable for ordinary M2304
  C1 prefill and decode.

## Results

| admission shape | warm aggregate input tok/s | outcome |
|:--|:--|:--|
| 7+7+7+7+4, M<=16128 | 4013 / 4091 / 4105 | M2304 tail returns to SDOT |
| 8+8+8+8, M<=18432 | 4932 / 4940 | stable, about +20.6% |
| 16+16, M<=36864 | 5302 / 5316 | best stable point, about +7.5% over 8-way |
| 32, M≈73724 | no completion | OOM requesting another 432 MiB |

The 16+16 service used about 51.4--52.8 GiB/GCD before the large forward.  The
single-batch experiment reached about 61.64 GiB PyTorch allocation with only
58 MiB free and is therefore rejected.  The selector ceiling is kept at
M=36864.

All successful arms completed every request.  Cross-round first-token hashes
were not bitwise exact, consistent with the existing BF16-CK numerical caveat;
this remains an experimental throughput oracle rather than a production
correctness checkpoint.

## Four-rank profile at the earlier 7+7+7+7+4 shape

The profiler reduced observed throughput from 4.39k to 4.03k, so times are
used only for component ranking.  Rank-max cumulative GPU times were:

- MHC pre-mix: 2.898 s.
- HIPBLASLt GEMMs: 2.793 s.
- M2304 SDOT tail gate+down: 2.250 s.
- RCCL: 1.393 s.
- sparse prefill attention: 1.242 s.
- CK MoE GEMMs: 1.081 s.
- indexer logits: 0.781 s.
- FP4-to-BF16 expansion: 0.733 s.
- MHC post: 1.391 s.

Moving to 8-way and then 16-way admission removes the small SDOT tail and
reduces full-model iteration count.  Further progress toward 10k must now
target the large-M MHC/projection/attention/collective path, not the already
closed I8 MFMA instruction schedule.
