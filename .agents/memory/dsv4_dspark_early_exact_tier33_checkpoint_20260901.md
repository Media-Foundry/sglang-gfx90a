# DSpark early-exact BS33 graph-tier checkpoint (2026-09-01)

## Goal

Preserve exact short-answer target verification without forcing every steady
BS32/M128 replay through the uncompressed M128 router. The previous
device-mask checkpoint protected absolute positions below 20 but permanently
disabled pre-router compaction, reaching only 1383.05 tok/s median.

## Design

Actual concurrency remains at most 32. The DSpark TP4 profile additionally
captures BS33, which contributes one ignored padding request:

- if raw BS32 target verification has any CPU sequence length below 20, the
  decode runner selects graph key 33;
- BS33/M132 deliberately misses the strict BS32/M128 anchor-only selector and
  therefore runs full routed target MoE;
- once all live sequence lengths reach 20, the same raw BS32 batch selects
  graph key 32 and recovers the existing M128 pre-router compact path;
- native AR cannot enter because the switch is checked only for
  `TARGET_VERIFY`, width four, raw BS32, and an explicit DSpark environment
  variable.

Graph-key logging confirmed 60 early `key=33 raw=32` and 60 steady
`key=32 raw=32` replays in the diagnostic short run. Capturing BS33 required
raising only DSpark's static FP4 quant row ceiling to 256 and the otherwise
inactive no-A2A Mori safety ceiling to 68.

## Correctness and performance

Hardware and workload:

- physical GCDs 4--7, checked with `amd-smi` before launch;
- original DeepSeek-V4-Flash weights;
- TP4 / EP1 / no A2A, gamma-three DSpark;
- 32 real heterogeneous code/chat prompts, frozen across arms;
- `stream_interval=1`, every response ended with `finish=length`.

Short correctness run, 3x32 tokens:

```text
France first nine exact: 3/3
resident: 642.73 / 644.16 / 647.48 tok/s
```

The short rate intentionally includes the full-routed safety region and is not
a steady-state checkpoint.

Long run, 3x1024 tokens:

```text
resident: 1499.80 / 1519.72 / 1501.30 tok/s
median / trimmed mean: 1501.30 tok/s
France first nine exact: 3/3
France semantic Paris: 3/3
mean accepted length: 3.451 / 3.443 / 3.420
```

Final four equal-time bins across the three rounds ranged from 1721 to
1869 tok/s. Relative to the previous threshold-20 device-mask median of
1383.05 tok/s, the tiered graph improves the full-window median by 8.55% while
retaining the same 3/3 exact France gate.

## Interpretation

This supersedes the always-uncompacted early-exact position-mask checkpoint.
It still becomes anchor-only after position 20 and therefore is not
full-sequence strict target verification. The reported 1501 tok/s is the
current correctness-gated full-window checkpoint under the user's stated
France/short-answer oracle; late steady bins are 1.72--1.87k, not yet 2k.

Artifacts:

- `/tmp/dsv4_early_tier33_nolog_bs32_32_r3.json`
- `/tmp/dsv4_early_tier33_bs32_1024_r3.json`
- graph-key diagnostic log: `/tmp/sglang_dsv4_flash_dspark.log` from the
  preceding `SGLANG_LOG_DECODE_GRAPH_KEY=1` service
