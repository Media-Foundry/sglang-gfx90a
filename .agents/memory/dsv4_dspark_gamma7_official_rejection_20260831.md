# DeepSeek V4 DSpark gamma-7 official-profile rejection on gfx90a

Date: 2026-08-31

## Motivation

The checkpoint README recommends vLLM DSpark with
`num_speculative_tokens=7`.  Earlier gfx90a experiments covered gamma 1, 2, 3,
and 5, but had no formal gamma-7 result.  This experiment closes that gap
instead of assuming that the checkpoint's recommended Blackwell profile also
fits four CDNA2 GCDs.

## Scope

- Original DeepSeek-V4-Flash checkpoint and bundled DSpark head.
- Physical `HIP_VISIBLE_DEVICES=4,5,6,7`.
- TP4 / EP1 / no A2A, BS32 maximum.
- `SPECULATIVE_DSPARK_BLOCK_SIZE=7`.
- 32 distinct concrete coding requests, 256 generated tokens.
- Fresh service and target/draft HIP graph capture.

## Correctness

The official France request passed the first-nine-token exact oracle and the
semantic Paris check.  All 32 coding requests returned 256 tokens with
`finish=length`.

## Result

```text
aggregate:          257.863 tok/s
resident BS32:      270.735 tok/s
scheduler:          269.540 tok/s
host step:          228.236 ms
mean accept length:   2.100
```

The short France workload had mean accept length `2.246` and resident
throughput `282.743 tok/s`.

Gamma 7 expands the full BS32 target verify tier to M256 and the draft region
to M224.  The extra accepted output is very small on real heterogeneous code
traffic, while the target and draft row counts grow sharply.  It is roughly
three times slower than the accepted gamma-1 checkpoint (about 769--783 tok/s
resident/scheduler, 70.9 ms host step, 1.76 accepted tokens).

## Decision

Reject gamma 7 for the four-gfx90a BS32 profile.  Keep gamma 1 as the service
baseline.  The README recommendation targets hardware with much higher FP4/FP8
row throughput; it is not an appropriate default for CDNA2.  Future gains must
raise M64 target row throughput or improve proposal quality without adding
target rows.

Artifacts:

```text
/tmp/dsv4_dspark_gamma7_official.log
/tmp/dsv4_gamma7_france.json
/tmp/dsv4_gamma7_code32_r1.json
```
