# DSpark position-wise acceptance and lean-graph rejection (2026-09-01)

## Scope

TP4/EP1/no-A2A on physical GCDs 4--7, original DeepSeek-V4-Flash weights,
gamma-three DSpark, 32 distinct code/chat requests, greedy generation and
`stream_interval=1`.  The accepted M128 anchor-only/pre-router compact profile
was unchanged.

The benchmark now records, for each equal-duration resident time bin, both
stream event rate and output tokens per event.  This separates target/draft
step throughput from speculative acceptance.

## Position-wise result

Two 1024-token rounds with the ordinary 32K-token pool reported:

```text
round 0 resident: 1432.84 tok/s, mean accept 3.583, France semantic true
  tok/s:       1030 1228 1410 1506 1555 1625 1615 1494
  events/s:     435  397  409  409  409  421  411  377
  tokens/event:2.37 3.09 3.44 3.68 3.80 3.86 3.93 3.96

round 1 resident: 1358.11 tok/s, mean accept 3.388, France semantic true
  tok/s:        920 1133 1290 1450 1457 1520 1540 1555
  events/s:     430  417  417  417  404  417  417  416
  tokens/event:2.14 2.72 3.10 3.48 3.61 3.65 3.70 3.74
```

The event rate is essentially flat.  The late-window 1.5--1.6k figures come
from acceptance rising toward the gamma-three maximum, not from kernels or the
scheduler warming up.  Therefore late-window throughput cannot be treated as
a general kernel-speed checkpoint.  Since anchor-only routed verification is
approximate, later high acceptance also needs repetition/quality auditing.

The second ordinary-pool round took 95.4 seconds group wall time because the
32K pool could not retain the whole heterogeneous 32x1024 workload and requests
were retracted/re-prefilled.  Its common resident window remains diagnostic,
but its group throughput is invalid for performance comparison.

Artifacts:

```text
/tmp/dsv4_gamma3_acceptance_bins_1024_2r.json
/tmp/dsv4_gamma3_fullreal_guard_256_5r.json
```

## Lean graph / larger KV experiment

The service was restarted with:

```text
CUDA_GRAPH_BS_DECODE="1 32"
MAX_TOTAL_TOKENS=49152
MEM_FRACTION_STATIC=0.96
```

This successfully reduced target capture from 25.81 to 5.71 seconds, draft
capture from 0.65 to 0.35 GiB, and allocated the 49,152-token pool.  The first
1024-token round retained France semantic correctness and measured 1424.74
resident tok/s; its final three bins were 1536/1621/1592 tok/s.

However, as requests completed, uncaptured BS31--2 target tiers fell back to
eager execution.  The tail became extremely slow and the second round reached
an eager M14 path that raised `hipErrorIllegalAddress`, terminating the
service.  A proposed model-graph admission guard was therefore removed: it is
a no-op when all 1--32 tiers are captured and exposes unsafe eager speculative
shapes when tiers are omitted.

Decision: reject the BS1/BS32-only service profile.  Retain all target/draft
tiers until the eager non-full-BS DSpark path is independently corrected, or
retain a dense set of drain tiers and prove every missing shape safe.  Do not
attribute the illegal address to hardware.

## Accepted full-tier 49K profile

Keeping every BS1--32 graph while raising only the memory budget worked:

```text
MAX_TOTAL_TOKENS=49152
MEM_FRACTION_STATIC=0.96

round 0: group 27.83 s, resident 1459.66 tok/s, accept 3.642, France true
round 1: group 28.60 s, resident 1442.98 tok/s, accept 3.609, France true
```

All 64 requests returned 1024 tokens with `finish=length`.  No retraction or
re-prefill tail appeared.  Late resident bins were 1.47--1.65k tok/s, while
the full-window two-round mean was 1451.32 tok/s.  Peak memory remained below
the 64-GiB/GCD limit and startup left roughly 2.8--2.9 GiB available after
graph capture.

This is accepted as a capacity/stability improvement, not as a target-kernel
speedup.  The TP4 BS32 script defaults are promoted to 49,152 tokens and 0.96
static memory while retaining all 1--32 decode graph tiers.

Artifact:

```text
/tmp/dsv4_gamma3_full_graph_49k_1024_2r.json
```

## AIter 1-MiB all-reduce geometry rejection

The existing default-off AIter geometry hook was rechecked on the accepted
49K/full-tier profile.  `AITER_GFX90A_AR_1M_BLOCKS=12` had previously reduced
the isolated 1-MiB TP4 all-reduce from about 62.36 to 44.82 us, but its two
real32 1024-token service rounds were:

```text
resident: 1477.89 / 1473.53 tok/s
mean:     1475.71 tok/s
France:   semantic true / false
```

The adjacent 80-block control was 1459.66/1442.98 (mean 1451.32), so the
candidate gained only 1.68% and reduced the France pass rate from 2/2 to 1/2.
Reject it: the isolated collective is too small a share of the full verify
step, and this synchronization-sensitive geometry does not clear correctness
or the 5% checkpoint gate.

Artifact: `/tmp/dsv4_gamma3_ar12_full_graph_49k_1024_2r.json`.

## M128 MHC geometry and M32 row-prefetch rejections

The exact wave64 M128 MHC kernel was re-swept before another service launch.
Rows-per-program 3/6/12/24 measured 60.62/59.61/81.70/116.19 us.  Rows=6 was
only 1.7% faster than the default rows=3, while larger values sharply reduced
parallelism.  All variants were bitwise exact, but the rows=6 budget is far
below the service noise floor, so the default remains rows=3.

The existing DSpark-profile-only M32 gate row-prefetch was then tested with
the accepted 49K/full-tier profile.  Two enabled service samples and the
adjacent disabled rollback were:

```text
enabled:  1472.94 / 1488.54 tok/s, mean 1480.74, France 2/2
disabled: 1427.45 / 1517.03 tok/s, mean 1472.24, France 2/2
delta:    +0.58% by two-round mean
```

The enabled runs were correct, but the result is smaller than the 5% commit
gate and well inside the acceptance-driven run-to-run spread.  Keep
`SGLANG_DSV4_GFX90A_M32_GATE_ROW_PREFETCH=0`; do not accumulate this switch
into the default profile merely because the isolated mechanism is exact.

Artifacts: `/tmp/dsv4_m32_rowprefetch_B2.json` and
`/tmp/dsv4_m32_rowprefetch_A2.json`.

The proposed compact-routed in-place anchor add was also implemented as a
strict child of the same M128 TARGET_VERIFY guard, then screened before a
service launch.  It kept the routed tensor at M32 and added it directly to
`shared_output[::4]`, replacing M128 zero + scatter + full add.  One hundred
random BF16 mutations were bitwise identical.  Captured graph medians were
12.23 us for the existing chain and 8.70 us for the candidate, only 3.53 us
per layer (about 0.15 ms per 43-layer step, below 0.2%).  The code was removed
without a service launch because it cannot materially close the 3.4% gap.

## Current gamma-three M128 rank-max localization

A diagnostic-only layer-20 realtime-marker service was captured on the same
49K/full-tier profile.  The marker logger deliberately synchronizes every 64
replays, so its HTTP rate (1139 tok/s over a short 256-token round) is not a
performance checkpoint.  France remained semantically correct and all 32
heterogeneous requests returned 256 tokens.

At the full BS32 M128 target tier, four-rank values were tightly grouped.  The
coarse layer span was about 1.43 ms/rank.  Fine MoE medians were approximately:

```text
router projection:       32.6--34.2 us
top-k:                   12.5--12.8 us
compact M32 routed MoE: 453.1--459.0 us
shared/routed join/add:   8.5--9.1 us
TP4 all-reduce tail:     73.3--80.2 us
```

Once requests began leaving the full tier, the observed layer fell to roughly
0.95 ms and routed work to 262--268 us.  The diagnostic confirms that closing
the conservative 1451.32-to-1500 gap requires about 60 us/layer and that only
the compact routed stage still has that local budget.  Existing MHC, router,
scatter/add, and geometry switches cannot supply it.

Artifact: `/tmp/dsv4_gamma3_m128_marker_256.json`; raw marker lines are in the
corresponding `/tmp/sglang_dsv4_flash_dspark.log` from the diagnostic service.

## Gate phase-fission and learned-Top5 rejections

An exact R1/W4 gate/up phase-fission oracle split the current combined kernel
into a gate projection writing FP32 `[32,6,512]` scratch and a separate up
projection consuming that scratch.  It preserved the production SDOT, scale,
DPP reduction and bounded-SwiGLU order.  All 100 input mutations and 1000 graph
replays were bitwise exact.  The current R2/W8 combined gate measured
246.07 us, while split grids 1664/2080/2496/3120 measured
405.38/391.33/381.52/375.13 us.  The extra phase launch, second LUT setup and
R1 task expansion dominate any occupancy improvement.  The oracle code was
removed; do not revisit projection fission without a single-launch primitive.

The only approximate target-only experiment with a sufficient theoretical
budget then retained Top-6 in the first three hash-router layers but dropped
the sixth learned expert in the remaining layers.  It was subordinate to the
existing TARGET_VERIFY/BS32/width4/M128/pre-router-compaction guard and never
affected AR.  Because invalid assignments leave fixed-slot partials unwritten,
the experimental runner also cleared its M32 partial buffer before reduction.

Three real32 256-token rounds were:

```text
resident: 1219.75 / 1166.01 / 1179.89 tok/s
accept:      2.965 /   2.874 /   3.007
France:      false /   false /   false
```

The learned-Top5 approximation both failed semantic correctness 0/3 and
reduced acceptance.  All model, runner and environment wiring was removed.
Artifact: `/tmp/dsv4_gamma3_m128_learned_top5_256_r3.json`.

## Gamma-three official full-block draft-attention recheck

The checkpoint's official full-block draft-attention mode was rechecked after
the CPU-length and live-`swa_loc` graph fixes, rather than relying on the older
pre-fix rejection.  It changed only the DSpark draft proposal; target anchor
math, original weights, gamma three, the 49K pool and all graph tiers stayed
unchanged.

Two real32 1024-token rounds produced:

```text
resident: 1413.26 / 1479.45 tok/s, mean 1446.36
accept:      3.560 /   3.634
France:      false /   false
```

The accepted control remains 1459.66/1442.98 (mean 1451.32, France 2/2).
Full-block attention therefore neither improves the long-window center nor
passes semantic correctness on the current stack.  Keep
`SGLANG_DSPARK_FULL_BLOCK_ATTN=0`.  The remaining acceptance opportunity needs
a fixed-input graph/eager proposal first-divergence audit rather than another
mask switch.

Artifact: `/tmp/dsv4_gamma3_fullblock_49k_1024_r2.json`.

## Gamma-three eager-draft parity diagnostic

To decide whether a graph/eager first-divergence tool was warranted, the
target verify graph was kept unchanged while only the draft transformer was
forced eager with `SGLANG_DSPARK_DISABLE_DRAFT_CUDA_GRAPH=1`.  The two real32
1024-token rounds were:

```text
resident: 1441.23 / 1391.25 tok/s, mean 1416.24
accept:      3.664 /   3.578
France:       true /   false
```

This is below the graph control's 1451.32 mean and does not improve semantic
stability.  Unlike the older gamma-five path, current gamma-three performance
is not limited by a large draft graph/eager acceptance divergence.  Keep the
draft graph enabled and do not build tensor-dump infrastructure solely for
this hypothesis.

Artifact: `/tmp/dsv4_gamma3_eagerdraft_49k_1024_r2.json`.

## NGRAM speculative control rejection

The same TP4/49K/full-tier target was started with the existing NGRAM worker,
gamma three and breadth one.  This preserves exact target verification but
runs the full M128 routed MoE instead of DSpark's anchor-only target path.  A
single decisive real32 1024-token round took 70.77 s group wall time:

```text
resident: 568.47 tok/s
accept:     1.803
France:     false
```

The second requested round was stopped after the first had already established
a roughly 2.55x resident regression versus the 1451.32 DSpark control.  Code
repetition did not supply enough NGRAM matches to offset full target cost.
Do not use standalone NGRAM as the four-GCD BS32 profile or mix this number
with DSpark throughput.

## Scheduler receive interval rejection

The launcher was temporarily given a default-inert environment passthrough so
the current DSpark profile could test `scheduler_recv_interval=2` while
retaining client `stream_interval=1`.  Two real32 1024-token rounds were nearly
identical:

```text
resident: 1428.594 / 1428.597 tok/s
accept:      3.533 /    3.512
France:       true /     true
```

This is stable but 1.6% below the 1451.32 control mean.  As with the older TP8
interval-four test, reducing scheduler receive frequency adds latency rather
than removing a critical host bottleneck.  The temporary launcher passthrough
was removed.

Artifact: `/tmp/dsv4_gamma3_recv2_49k_1024_r2.json`.

## Confidence-budgeted gamma-three M96 rejection

A strict compact-ragged M96 experiment kept gamma three but forced the planner
to verify only two thirds of the 96 draft rows.  The 32 mandatory anchors plus
64 confidence-selected drafts fit an exact 96-token graph.  A temporary
TARGET_VERIFY/BS32/width4/M96 guard derived live anchor rows from
`qo_indptr_device`, retained routed MoE only on those anchors, and left AR
unreachable.  Focused guard tests passed 4/4 before service launch.

The first real32 256-token screening round was decisive:

```text
resident: 992.34 tok/s
accept:     2.590
France:     semantic Paris
```

The M96 graph savings did not offset the valuable removed drafts and the
non-compacted M96 router/top-k work.  This is far below the current gamma-three
short-window rate, so no long run was made.  All M96 model/env/test wiring was
removed.  Artifact: `/tmp/dsv4_gamma3_m96_budget66_256_r1.json`.
