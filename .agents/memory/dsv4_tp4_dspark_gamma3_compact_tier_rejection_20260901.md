# DeepSeek-V4-Flash TP4 DSpark gamma=3 compact tier rejection (2026-09-01)

## Experiment

One service captured compact target verification tiers M64/M80/M96/M112/M128
on physical GCDs 4--7. Gamma remained three; draft graph was disabled so all
tiers shared the same eager draft implementation. Fixed draft-budget fractions
selected each exact tier. The workload was the standard 32 distinct code/chat
token-ID prompts, 256 generated tokens, stream interval one.

A temporary, strictly speculative guard routed only the 32 anchor rows through
the target MoE for all compact tiers. It was removed after the sweep.

## Results

| Tier | Resident BS32 tok/s | Mean accepted length | France oracle |
|---|---:|---:|---|
| M64 | not retained (harness stopped immediately) | -- | fail |
| M80 | 842.81 | 2.391 | fail |
| M96 | 890.84 | 2.598 | fail |
| M112 | 981.46 | 2.828 | fail |
| M128 | 978.41 | 2.942 | fail |

All recorded M80--M128 requests completed 32/32 at 256 tokens with
`finish=length`, but every tier failed the France first-nine exact oracle.
M128 full budget also failed, proving this was not merely over-aggressive
confidence pruning. Compact-ragged target verification plus anchor-only routed
compute is not semantically equivalent to the accepted static gamma=3 path.

No tier exceeded the recent static gamma=3 control (~1045.5 tok/s), even if the
correctness failure were ignored. The route is rejected and the experimental
generic compact anchor guard was reverted.

Evidence:

- `/tmp/dsv4_gamma3_compact_m80.json`
- `/tmp/dsv4_gamma3_compact_m96.json`
- `/tmp/dsv4_gamma3_compact_m112.json`
- `/tmp/dsv4_gamma3_compact_m128.json`
