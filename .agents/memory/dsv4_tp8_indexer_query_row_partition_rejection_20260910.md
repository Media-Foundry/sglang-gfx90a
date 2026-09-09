# TP8 C4 indexer query-row partition rejection (2026-09-10)

## Proposal

Partition the strict DSpark M128 C4-indexer query rows across TP8 ranks.  Each
rank would compute 16 complete rows (all 64 index heads), then exchange only
the resulting Top-512 logical int32 IDs.  Each rank would map those logical IDs
through its local page table.  This avoids the previously rejected head-shard
design's FP32 score all-reduce.

## Component evidence

Single-GCD BF16 projection timings for the replicated indexer projections were:

| projection | M16 | M32 | M64 | M128 |
|---|---:|---:|---:|---:|
| `wq_b`, 1024x8192 | 26.37 us | 26.01 us | 30.94 us | 36.06 us |
| `weights_proj`, 4096x64 | 29.96 us | 30.49 us | 30.30 us | 29.92 us |

Thus M128 to M16 saves only about 9.7 us in `wq_b` and nothing in
`weights_proj`; launch and weight-read cost dominate at these small row counts.

The exact communication payload is 16x512 int32 per rank, reconstructed as
128x512 int32 on every rank (256 KiB).  A graph-replay TP8 oracle measured:

| backend | rank-max median | rank-max trimmed mean | exact |
|---|---:|---:|---:|
| RCCL int32 all-gather | 224.79 us | 220.22 us | yes |
| AIter float32 bitcast all-gather | 37.05 us | 37.09 us | yes |

AIter's public integer dtype guard was preserved.  The oracle bit-casts int32
storage to float32 because all-gather is byte-preserving, then validates the
result against exact int32 IDs.  This is only an oracle technique, not a
production change.

## Critical applicability result

The current short-context C32 benchmark has effective C4 lengths below the
Top-512 threshold.  Production therefore skips C4 logits/Top-K entirely.  A
query-row partition cannot improve the current 2k decode objective even if its
long-context communication were free.

For long contexts, the measured projection saving is already smaller than the
37 us optimized exchange before packing, row reconstruction, or local physical
mapping.  Do not wire this design into production.  Reconsider only if a future
long-context profile shows a much larger paged-logits term and an exchange can
be fused into an existing TP boundary.

Reproducible oracle:

`scripts/rocm/bench_dsv4_tp8_indexer_ids_allgather.py`

Raw log: `/tmp/tp8_indexer_ag_full.log`.

