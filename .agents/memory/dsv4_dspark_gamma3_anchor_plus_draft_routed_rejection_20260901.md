# DSpark gamma3 anchor-plus-draft routed rejection (2026-09-01)

Tested the previously unmeasured fixed-row combination for BS32 gamma-three:
retain the full M128 attention/shared path, but compact `[anchor, draft0]` from
each four-row request before router/TopK and execute complete Top-6 routed MoE
as physical M64. This differs from earlier confidence-M96 and partial-expert
tests: draft0 received all six routed experts.

The experiment used original weights, TP4/EP1, GCDs 4--7 and the 32-request
heterogeneous code manifest. Service and graph capture succeeded, but the
first correctness round failed the France exact gate:

```text
expected first nine: 671 6102 294 8760 344 2619 51119 42499 1
observed first nine: 671 6102 294 8760 344 2619 51119 13510 778
```

Testing stopped immediately; no throughput result is accepted. The temporary
row-count selector and M64 carrier generalization were removed. Keep only the
existing anchor-only pre-router experiment (default off). Do not infer that
adding mathematically fuller draft routed rows preserves the accepted
anchor-only DSpark trajectory: the verifier/bonus path is sensitive to those
logits.
