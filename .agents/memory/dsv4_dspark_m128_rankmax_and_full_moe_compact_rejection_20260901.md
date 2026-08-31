# DSpark gamma-3 M128 rank-max trace and full-MoE compact rejection (2026-09-01)

## Clean trace protocol

The accepted TP4/EP1 gamma-3 profile (`fc6bd83823`) was launched on physical
GCDs 4--7 with layer-20 realtime markers.  Startup readiness was determined
from the process log and TCP listener; no HTTP/curl polling was used.  Two
finite rounds of 32 heterogeneous code requests at 128 generated tokens were
then issued with `stream_interval=1`.  Marker instrumentation is diagnostic
and reduced resident throughput to about 989/1008 tok/s, so these rounds are
not performance checkpoints.

For the early resident-BS32 samples, four-rank maxima were approximately:

```text
MHC entry                    94--95 us
attention prepare          278--279 us
attention core              72--118 us
attention output/collective 199--201 us
FFN entry                   114--116 us
MoE                         579--619 us

MoE detail:
router                       32--33 us
Top-K                        13 us
routed expert              445--475 us
join/add                      5 us
TP4 all-reduce               79--91 us
```

Later samples include batch-tier drain and must not be mixed into the BS32
rank-max budget.  Raw artifacts are:

```text
/tmp/dsv4_gamma3_compact_marker.log
/tmp/dsv4_gamma3_compact_marker_bench.json
```

## Full-MoE anchor compaction screen

The routed branch already compacts fixed gamma-3 M128 rows `0::4` to M32.  A
strictly guarded, default-off prototype additionally:

1. ran the shared expert on the same 32 anchor rows;
2. kept routed+shared partials compact through the TP4 all-reduce;
3. scattered the reduced M32 result back to rows `0::4` of a zero M128 output.

The path required gfx90a + TARGET_VERIFY + BS32 + width four + M128 and was
therefore unreachable from native AR.  Unit tests for the existing selector
passed before the service screen.

The first France correctness gate failed immediately:

```text
[671, 6102, 574, 294, 8760, 1, 344, 2619, 574]
semantic Paris: false
first-nine exact: false
```

No throughput result was accepted.  The prototype was removed completely.
This agrees with the earlier gamma-1/M64 shared-anchor rejection: draft rows
need the dense shared-expert contribution even when routed-expert
approximation is tolerated.  Do not retry shared-anchor compaction without a
new target-verification semantics argument.

## Decision

- Keep routed-only M128 pre-router compaction.
- Keep the full M128 shared branch and full M128 TP4 all-reduce.
- Optimize the measured routed/prepare/attention-output regions without
  dropping the shared contribution from draft rows.

