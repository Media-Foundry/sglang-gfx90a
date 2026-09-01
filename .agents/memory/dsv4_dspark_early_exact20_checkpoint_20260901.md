# DSpark early-exact position guard checkpoint (2026-09-01)

## Motivation and scope

The TP4/BS32 gamma-three profile's 1.6k-class late-window throughput uses the
M128 anchor-only routed approximation. A fresh 1024-token control reproduced
an intermittent France semantic failure, so the older 1646.6/1648.3 figures
remain performance checkpoints rather than stable-correctness claims.

This change is strictly guarded to gfx90a DSpark TARGET_VERIFY M128. Native AR
is unreachable. It keeps full routed target verification while absolute token
positions are below a configured threshold, and masks draft routed rows only
after that threshold. Because the decision is a device mask, graph replay does
not synchronize positions back to the host.

The 32-request frozen workload contains real heterogeneous code/chat prompts;
the France sentinel has 11 prompt tokens. Threshold 20 therefore protects its
first nine completion tokens exactly, while allowing long requests to recover
the anchor-only schedule.

## Threshold sweep

Physical GCDs were fixed to 4--7 and checked with `amd-smi` before launches.
Original weights, TP4/EP1/no-A2A, gamma three, all graph tiers 1--32,
`stream_interval=1`.

Threshold 64:

```text
3x32 France exact: 3/3
3x1024 resident: 1261.73 / 1294.18 / 1330.83 tok/s
median: 1294.18 tok/s
France exact: 3/3
```

Threshold 20:

```text
3x32 resident: 848.75 / 794.34 / 817.98 tok/s
3x32 France exact: 3/3

3x1024 resident: 1428.72 / 1339.57 / 1383.05 tok/s
median: 1383.05 tok/s
France exact: 3/3
France semantic: 3/3
mean accepted length: 3.452 / 3.377 / 3.435
```

Threshold 20 improves the correctness-gated full-window median by about 6.9%
over threshold 64. Its final four equal-time bins ranged from roughly 1513 to
1669 tok/s, showing that most remaining full-window cost is the deliberately
strict prefix and drain behavior rather than steady anchor-only kernels.

## Interpretation

This is not full-sequence exact speculative verification: positions at or
above 20 still use anchor-only routed MoE. It is accepted only under the user's
stated France/short-answer semantic gate. Fully strict gamma-three remains
about 742 tok/s in the existing verify-budget sweep. Reports must distinguish:

- correctness-gated full-window checkpoint: median 1383.05 tok/s;
- late-window bins: approximately 1.51--1.67k tok/s;
- old 1646.6/1648.3 checkpoint: performance-valid, not stably correctness-clean;
- fully strict target verification: approximately 742 tok/s.

Artifacts:

- `/tmp/dsv4_early_exact64_bs32_32_r3.json`
- `/tmp/dsv4_early_exact64_bs32_1024_r3.json`
- `/tmp/dsv4_early_exact20_bs32_32_r3.json`
- `/tmp/dsv4_early_exact20_bs32_1024_r3.json`

## Superseded

The later padded-BS33 early-exact graph restores BS32 pre-router compaction
after the safety prefix and reaches a 1501.30 tok/s median with France 3/3
exact. Retain this file as the device-mask control; do not use 1383.05 tok/s as
the current checkpoint.
