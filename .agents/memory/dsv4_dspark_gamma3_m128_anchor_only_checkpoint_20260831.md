# DSV4 DSpark gamma-three M128 anchor-only routed checkpoint

Date: 2026-08-31

## Scope

The TP4 BS32 DSpark profile now defaults to gamma three. Static target verify
lays out each request as:

```text
[anchor, draft_0, draft_1, draft_2] * 32 = M128
```

The anchor row controls target acceptance and remains a full target-model row.
The three draft rows only provide later/bonus logits. The optimization keeps
all six routed experts on each anchor row, replaces draft-row top-k IDs with
the established `-1` sentinel, and zeroes draft routed output before shared
expert add. Draft rows retain shared experts, attention, MHC, norm, and every
other model component. Original checkpoint weights are unchanged.

The selector requires every condition below:

```text
SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED=1
gfx90a
ForwardMode.TARGET_VERIFY
ForwardBatch.batch_size == 32
spec_info.num_tokens_per_req == 4
hidden_states.shape == [128,4096]
```

Native AR, gamma one/two, every other graph tier, and other architectures are
ineligible. The older gamma-one M64 selector remains as an explicit rollback
profile and is inert under the gamma-three default.

## Motivation from graph markers

The accepted gamma-one M64 checkpoint measured about `1.21 ms/layer` at layer
20. Routed FP4 occupied roughly `449 us` of a `552 us` MoE span. Gamma one
already removed odd draft routed rows, so its remaining routed work represented
32 exact anchors. Increasing gamma while preserving exactly the same 32 routed
anchor rows lets accepted length grow without multiplying the largest weight
scan.

Gamma two improved only 1.55% because its extra attention cost nearly cancelled
the accepted-length gain. Gamma three benefits from the existing M128 decode
geometry and crossed the service checkpoint threshold.

## Correctness

The model-side unit tests cover positive reachability and negative guards for
native/non-target forward mode, BS16, speculative width three rather than four,
M96 rather than M128, and missing speculative metadata.

Across the B-A-B runs and the final default-profile validation:

```text
France first nine token IDs: exact in every round
France semantic answer:      Paris in every round
32 varied code requests:     every request generated 256 tokens
finish reason:               length for every request
```

An AR negative-control service deliberately set the M128 environment variable
to one. Since the forward mode was native AR, the selector remained
unreachable:

```text
France first nine: exact
France semantic:   Paris
32 varied requests: all 64 tokens, finish=length
spec accept fields: absent
resident rate: 716.170151 tok/s (reachability evidence only)
```

## B-A-B performance

All arms used physical GCDs 0--3, 32 different concrete code prompts, 256
generated tokens, and `stream_interval=1`.

```text
B1 gamma3 M128 anchor-only:
  868.092187, 909.094433, 877.648859
  median 877.648859 tok/s

A gamma1 M64 anchor-only:
  818.621861, 826.172119, 836.045818
  median 826.172119 tok/s

B2 gamma3 M128 anchor-only:
  885.161337, 821.512183, 874.358788
  median 874.358788 tok/s

center(B1,B2) = 876.003824 tok/s
gain vs A      = +6.0328%
```

Gamma-three accepted length was approximately `2.27--2.53`, versus
approximately `1.63--1.67` for gamma one. Candidate host steps were generally
`62--69 ms`, versus `56--58 ms` for control; the accepted-length increase won.

## Final default-profile validation

The service was restarted without manual gamma or M128 environment overrides.
Its command line contained `--speculative-dspark-block-size 3`, proving the
script default was active. Three final rounds reported:

```text
resident: 839.532754, 899.224659, 887.837312 tok/s
median:   887.837312 tok/s
accept:   2.268430, 2.384941, 2.451127
```

All correctness gates above passed again.

## Decision

Adopt gamma three plus strict M128 anchor-only routed MoE as the TP4 BS32
`start-dspark` default. It is a speculative quality/performance tradeoff, not
native AR and not bitwise full-target bonus inference, even though the tested
France trajectory stayed exact. Set either of these for rollback:

```text
SPECULATIVE_DSPARK_BLOCK_SIZE=1
SGLANG_DSV4_GFX90A_DSPARK_M128_ANCHOR_ONLY_ROUTED=0
```

The new stable checkpoint is about 876--888 resident tok/s, still below the
1.5k objective. The next major target is M128 attention/indexer/compressor,
because routed work is already reduced to the 32 exact anchors.

Artifacts:

```text
/tmp/dsv4_gamma3_anchor_probe.json
/tmp/dsv4_gamma3_anchor_b1.json
/tmp/dsv4_gamma1_control2.json
/tmp/dsv4_gamma3_anchor_b2.json
/tmp/dsv4_gamma3_default_final.json
/tmp/dsv4_ar_m128_negative.json
```

