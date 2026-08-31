# DSV4 DSpark gamma-1 M64 draft top-1 routed rejection (2026-08-31)

## Scope and isolation

This experiment modified only the already guarded speculative target-verify
path.  The candidate required all of:

- `SGLANG_DSV4_GFX90A_DSPARK_M64_ANCHOR_ONLY_ROUTED=1`;
- gfx90a;
- `ForwardMode.TARGET_VERIFY`;
- scheduler batch size 32;
- `spec_info.num_tokens_per_req == 2`;
- hidden-state shape `[64, 4096]`.

Native AR cannot satisfy this guard.  No common MoE kernel selector, weight,
attention path, or AR launch profile was changed.

The measured checkpoint keeps the full routed MoE on even anchor rows and
removes routed MoE entirely from odd draft/bonus rows.  This candidate instead
kept the highest-ranked routed expert on every odd row and masked routed slots
1--5.  Shared experts remained enabled for both rows.

## Placement

A fresh `amd-smi process --general --sort-by-pid` showed that an unrelated BIO
job still owned 934.4 MiB on physical GCD 4.  The candidate service therefore
used physical GCDs 0--3 for a directional rejection run; the foreign process
was not killed.  These values must not be mixed with clean 4--7 checkpoints.

## Result

The strict France oracle failed immediately:

```text
expected first nine: established exact checkpoint
candidate first ids: [671, 6102, 294, 8760, 17, 2619, 51725, 16540, 6279]
```

One diagnostic round with mismatch recording enabled was worse both
semantically and operationally:

```text
France first-nine exact: false
France semantic Paris:   false
all 32 code requests:    256 tokens, finish=length
resident BS32:           553.571649 tok/s
scheduler decode:        556.931879 tok/s
mean accept length:      1.307572
mean accept rate:        0.304509
mean host step:          61.614584 ms
```

The retained routed slot changes the approximate bonus trajectory without
recovering acceptance, while its extra routed work destroys the speed advantage
of the shared-only odd rows.  The previously accepted anchor-only checkpoint
was around 802 tok/s at the matched-card center and retained France/Paris.

Artifact:

```text
/tmp/dsv4_dspark_slots1_realcode_allow.json
```

## Decision

**Reject draft top-1 routed retention.**  The sweep implementation was fully
reverted, so production remains at the strict DSpark-only anchor-routed / odd
shared-only checkpoint.  Do not test top-2 through top-5: top-1 already loses
both the semantic gate and roughly 31% resident throughput versus the accepted
checkpoint, and additional slots can only add routed work and weight scans.

