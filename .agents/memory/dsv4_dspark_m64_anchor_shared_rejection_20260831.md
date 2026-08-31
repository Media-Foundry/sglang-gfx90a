# DSV4 DSpark M64 anchor-only shared-expert rejection (2026-08-31)

## Candidate

The accepted speculative checkpoint already keeps routed MoE only on the even
anchor rows of gamma-one target verification.  This follow-up also evaluated
the shared expert only on those even rows:

```text
hidden[0::2] -> shared expert M32 -> scatter into even rows
hidden[1::2] -> zero shared contribution
```

The experiment was subordinate to the existing strict selector:

```text
gfx90a && TARGET_VERIFY && BS32 && width=2 && hidden=[64,4096]
```

It therefore never selected native AR.  It changed no weight, attention
backend, common shared-expert kernel, or AR profile.  The implementation was
fully reverted after rejection.

## Placement and correctness

The external BIO job still owned physical GCD 4, so the directional service
used physical GCDs 0--3 after an `amd-smi` process scan.  The foreign job was
not disturbed.

The strict France gate failed immediately:

```text
candidate first ids: [671, 6102, 574, 294, 8760, 666, 344, 666, 344]
France first-nine exact: false
France semantic Paris:   false
```

All 32 varied code requests still returned 256 tokens with `finish=length`,
but the decoded trajectories were visibly repetitive and therefore do not
pass the quality contract.

## Performance

One mismatch-enabled diagnostic round reported:

```text
aggregate:             766.115872 tok/s
resident BS32:         762.887146 tok/s
scheduler decode:      862.589958 tok/s
mean accept length:    1.749544
mean accept rate:      0.746409
mean host step:        58.157 ms
```

The resident result is below the approximately 802 tok/s matched-card center
of the accepted odd-shared checkpoint.  M32 shared compute plus contiguous
gather/scatter also did not translate into a service win large enough to
justify the severe bonus-logit degradation.

Artifact:

```text
/tmp/dsv4_dspark_anchor_shared_realcode_allow.json
```

## Decision

**Reject anchor-only shared expert.**  Odd draft rows require the shared-expert
contribution to preserve a useful bonus trajectory.  Keep full M64 shared
expert compute while retaining the accepted anchor-only routed approximation.

