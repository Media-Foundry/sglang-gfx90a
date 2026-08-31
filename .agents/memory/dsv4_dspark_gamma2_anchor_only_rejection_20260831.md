# DSV4 DSpark gamma-two anchor-only routed rejection (2026-08-31)

## Baseline revalidation and measurement pitfall

The accepted gamma-one M64 checkpoint was revalidated on physical GCDs 0--3
with 32 different concrete code prompts, 256 generated tokens per request, and
the France/Paris oracle.  With the harness's comparable `stream_interval=1`
its three resident rounds were:

```text
795.775978, 829.778705, 807.178262 tok/s
median 807.178262 tok/s
```

All requests finished with length 256 and France was first-nine exact plus
semantic Paris.  Scheduler throughput was `602.36/628.39/613.33 tok/s`, with
host steps `56.989/54.628/55.990 ms`.

Using `stream_interval=32` had misleadingly reported only about 630 tok/s.
Resident windows are reconstructed from client streaming timestamps, so coarse
32-token chunks cannot be compared with the historical interval-one data.
This was a measurement artifact, not a kernel regression.

A separate NUMA1 binding test for GCDs 4--7 was also negative: resident median
fell from about 647.9 (unbound, interval 32) to 613.7 tok/s, while host step
remained around 54--55 ms.  Do not add NUMA binding as a DSpark default from
that experiment.

Artifacts:

```text
/tmp/dsv4_dspark_gcd03_interval1.json
/tmp/dsv4_dspark_clean_baseline_realcode.json
/tmp/dsv4_dspark_numa1_realcode.json
```

## Current gamma-one M64 layer budget

The layer-20 graph-replay realtime marker on the accepted path reported a
rank-max span of about `1.21 ms/layer`.  A representative rank decomposed to:

```text
coarse deltas: 90.88, 1.44, 264.64, 67.52, 138.88, 98.88, 552.00 us
MoE router:      38.24 us
MoE top-k:       16.32 us
routed FP4:     449.44 us
join/add:         9.44 us combined
TP4 AR/tail:     34.56 us
```

The routed stage remains the largest single component, but gamma-one already
removes every odd draft routed row; its remaining work is the 32 exact anchors.

## Gamma-two candidate

Static gamma-two target verification lays out M96 as 32 contiguous request
rows of `[anchor, draft_0, draft_1]`.  The candidate retained the full target
routed MoE on rows `0::3`, replaced both draft rows' top-k IDs by the existing
`-1` sentinel, and zeroed their routed outputs before shared-expert add.  Shared
experts and every non-routed component remained unchanged.

The selector required every condition below and therefore could not affect AR:

```text
gfx90a
ForwardMode.TARGET_VERIFY
batch_size == 32
spec_info.num_tokens_per_req == 3
hidden_states == [96,4096]
explicit M96 environment opt-in
```

Unit tests covered the positive selector and negative AR, BS16, width-two,
M64, and missing-metadata cases.

## B-A-B service result

All arms used physical GCDs 0--3, 32 real varied code prompts, 256 generated
tokens, and `stream_interval=1`.

```text
B1 gamma2 M96 anchor-only: 828.094, 873.216, 826.643; median 828.094
A  gamma1 M64 anchor-only: 796.170, 828.035, 821.245; median 821.245
B2 gamma2 M96 anchor-only: 821.487, 874.729, 839.887; median 839.887

center(B1,B2) = 833.990 tok/s
gain vs A      = +1.552%
```

Gamma two raised mean accepted length from approximately `1.61` to
`1.98--2.06`, but raised host step from approximately `55--57 ms` to
`60--67 ms`.  The effects almost exactly cancelled.

France first-nine IDs differed because the bonus trajectory is approximate,
but every round remained semantic Paris.  All 32 varied code requests produced
256 tokens with `finish=length`.  This satisfies the relaxed semantic probe but
does not justify a below-5% checkpoint.

Artifacts:

```text
/tmp/dsv4_gamma2_anchor_b1.json
/tmp/dsv4_gamma1_anchor_a.json
/tmp/dsv4_gamma2_anchor_b2.json
/tmp/dsv4_dspark_gamma2_anchor_allow.json
```

## Decision

**Reject gamma-two anchor-only routed as a production default.**  The measured
center is only 1.55% above gamma one, far below the 5% checkpoint threshold and
far from the 1.5k objective.  The experimental implementation was fully
reverted; production remains the strict gamma-one M64 DSpark-only selector.

