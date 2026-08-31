# DSV4 TP4/M64 attention-tail knockout and row-chunk rejection (2026-08-31)

## Scope

- Original tensor shapes for TP4/M64: attention output `[64,16,512]`, grouped
  `wo_a` input `[64,2,4096]`, weights `[2,1024,4096]`, flattened intermediate
  `[64,2048]`, local `wo_b` weight `[4096,2048]`, TP4 all-reduce payload 512 KiB.
- Physical devices `HIP_VISIBLE_DEVICES=4,5,6,7`.
- Registered AIter custom all-reduce, rank-max timing, seven-round interleaving.
- 100 input mutations and 1000 HIP graph replays passed exact intermediate,
  partial and reduced-output comparisons.

The existing `scripts/rocm/bench_dsv4_tp4_m32_attention_tail_knockout.py` now
accepts `--tokens 32|64` and includes local `wo_b` row-chunk and `wo_a` layout
lower bounds. No production selector or model path changed.

## A/B/C/D knockout

Profiles are:

- A: oracle inverse RoPE + production-shape `wo_a` + `wo_b` + AR;
- B: externally prepared exact `wo_a` alias + `wo_b` + AR;
- C: externally prepared exact `wo_b` partial alias + AR;
- D: exact reduced alias and empty graph.

Representative seven-round trimmed rank-max values were:

| profile | time |
|---|---:|
| A | 160.429 us |
| B | 74.408 us |
| C | 43.818 us |
| D | 1.005 us |

This gives `wo_b` about 30.59 us and registered AR about 42.81 us.

Important caveat: A uses a simple Torch inverse-RoPE spelling for a portable
oracle, not the production fused in-place RoPE kernel. Therefore A-B = 86.02
us is **not** a production inverse+`wo_a` budget and must not be used to claim
an 86-us optimization opportunity. The independent pure-`wo_a` graph below is
the valid local projection measurement.

## `wo_a` layout lower bound

The exact M64 grouped projection was compared as current-style einsum versus
strided batched matmul on the same precomputed inverse-RoPE input:

```text
einsum: 42.102 us
bmm:    42.058 us
delta:  -0.045 us
```

The complete BF16 output was bitwise exact. Rewriting the framework operation
as `bmm` is neutral and should not be connected to production.

## `wo_b` row-chunk lower bound

The complete M64 `wo_b` producer was compared with two sequential M32 row
chunks before implementing any new communication protocol:

```text
whole M64: 35.067 us
2 x M32:   61.633 us
penalty:   +26.566 us
max abs:   0.00781250 (not bitwise exact)
```

The ideal K=2 overlap could hide at most half of the roughly 42.8-us AR, about
21.4 us. The measured producer penalty already exceeds that impossible ideal,
before adding publication, synchronization or extra collective entry costs.
Reject row-chunk `wo_b -> AR` pipelining at M64.

## Decision

- Do not replace `wo_a` einsum with `bmm`.
- Do not implement M64 row-chunk all-reduce or extend the existing M32 peer
  publication protocol.
- A future output-tail kernel must remove a real intermediate/launch boundary
  without persistent full-grid barriers and must demonstrate at least 15 us
  per-layer standalone savings before service integration. Existing M1/M32
  persistent-fusion service regressions make this a low-priority route.

Raw logs:

- `/tmp/dsv4_tp4_m64_attention_tail_knockout.log`
- `/tmp/dsv4_tp4_m64_attention_tail_rowchunk.log`
- `/tmp/dsv4_tp4_m64_attention_tail_woa_layout.log`

