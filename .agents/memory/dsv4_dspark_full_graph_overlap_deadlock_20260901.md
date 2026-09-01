# DSpark full-graph overlap oracle rejection (2026-09-01)

## Question

Could a second cohort's roughly 10 ms draft graph be hidden under the current
roughly 58 ms target-verify graph, providing the multi-millisecond structural
gain still needed after the M128 CK checkpoint?

## Diagnostic design

A default-off DSpark-worker-only prototype:

1. ran the normal proposal;
2. cloned every proposal tensor consumed later by target verification,
   acceptance, confidence, and observability;
3. replayed the same draft graph on a private HIP stream, discarding output;
4. concurrently launched target verification on the main stream;
5. joined with a device event before the next step, without host synchronize.

The draft and target runners have independent graph staging buffers and KV
pools, but both graphs contain TP4 custom-all-reduce/collective operations.
The prototype was never enabled in native AR and no production result depended
on it.

## Result

The service loaded and captured all graph tiers normally, but its first BS1
France request made no progress for more than 30 seconds. Four ranks retained
about 60--61 GiB/GCD, no HTTP completion arrived, and no Python exception was
logged. This is a device-side collective ordering/epoch deadlock, not an OOM.
The request and service were stopped; GCD 4--7 allocations were released.

The complete prototype was removed. No shadow-overlap environment variable,
proposal cache, extra stream, or clone path remains in source.

## Implication

Do not overlap two complete TP graphs on independent streams. Their per-layer
collectives must preserve one global rank order. A viable dual-cohort design
must instead isolate compute-only regions and schedule collectives through one
ordered communication stream, for example:

```text
target cohort compute segment
        || draft cohort compute-only segment
        -> one ordered TP collective queue
```

The next oracle should begin below the full graph boundary: overlap a draft
attention/MoE compute segment that contains no custom AR with a target routed
or sparse-attention segment, then measure whether at least 4--6 ms remains
hideable after enforcing the original collective sequence.

Artifact: `/tmp/sglang_dsv4_shadow_overlap2.log`.
