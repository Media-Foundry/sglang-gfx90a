# DSpark gamma-3 M128 static routed-layer skipping rejection (2026-09-01)

To test the only approximate routed direction with enough absolute budget to
matter, a default-off quality screen suppressed the anchor routed branch in a
static set of learned layers while retaining the complete M128 shared expert,
attention, MHC and TP4 collective.  The selector inherited the existing
gfx90a TARGET_VERIFY/BS32/width-four/M128 pre-router guard, so native AR was
unreachable.  The implementation only replaced the selected layer's Top-6 IDs
with `-1`; it did not change weights or other rows.

Both screens failed the first France round:

```text
skip layers 8,16,24,32,40:
671 64780 64780 118235 127269 114560 35784 35784 60900

skip only layer 20:
671 11111 14 11111 14 305 223 20 289
```

The single-layer failure is decisive: target-verifier logits are highly
sensitive to every learned routed residual, even though draft rows already use
the accepted anchor-only routed approximation.  The selector was removed and
no throughput result is accepted.  Do not pursue static routed-layer knockout
without a trained compensation/distillation mechanism.

