# DSpark M128 attention-output exact-oracle rejections (2026-09-01)

## `wo_b -> TP4 all-reduce` row pipeline bound

The existing attention-tail oracle was extended to M128 during the 1-MiB AR
sweep.  It directly measured:

```text
whole M128 wo_b:       about 42.3 us
two sequential M64:   about 66.3 us
producer penalty:     about 24.0 us
default 1-MiB AR:     about 62.4 us
tuned 1-MiB AR:       about 44.8 us
```

With two chunks, an impossible zero-overhead pipeline can hide at most half of
the AR: 31.2 us with the default collective or 22.4 us with the tuned one.
The measured producer penalty is already 24.0 us, before adding a second
collective entry, publication, epoch tracking or graph nodes.  Thus the tuned
case is provably slower and even the default case has an upper bound of only
about 7 us/layer.  Do not implement the row-chunk collective pipeline.

## TP4 M128 output-N projection bundle

`scripts/rocm/bench_dsv4_tp4_output_n_projection_ag.py` now accepts M128 and
uses a deterministic BF16 fallback when the old layer dump is unavailable.
It compares the four production projections (`N=1536/2048/512/64`) against
output-N sharding plus registered TP4 all-gather.

Full N4160 bundle, 128 mutations, four-rank ABBA:

```text
separate projections: 146.440 us
bundle + all-gather:   131.634 us
saving:                 14.806 us (below 30-us gate)
exact:                  false
mismatches:             124/128
max abs:                0.015625
max relative L2:        2.40e-5
```

Only the N64 index-weight segment differed; qkv/core/index-kv remained exact.
The local GEMM-to-AG slice was exact, so the difference again comes from GEMM
shape/reduction association.

Hybrid N3584 bundle (qkv+core only; index paths independent):

```text
separate projections: 147.235 us
hybrid bundle:        182.740 us
change:               -35.504 us
exact:                true
```

The exact hybrid loses badly, while the non-exact full bundle saves only
14.8 us before its consumer-completion barrier.  This is much weaker than the
old M32 micro result and cannot justify another service integration.

Artifacts:

```text
/tmp/dsv4_m128_output_n_full.log
/tmp/dsv4_m128_output_n_hybrid.log
```

## Decision

- Reject M128 `wo_b` row-chunk pipelining.
- Reject both full and hybrid M128 output-N bundles.
- Preserve independent attention producers; target larger routed/acceptance
  structure rather than projection concatenation.

