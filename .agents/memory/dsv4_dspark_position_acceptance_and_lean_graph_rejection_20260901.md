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
