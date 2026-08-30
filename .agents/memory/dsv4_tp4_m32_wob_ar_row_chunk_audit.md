# DSV4 TP4 M32 `wo_b` + AR row-chunk pipeline audit

Date: 2026-08-30

## Existing measurements

The real layer-20 TP4 attention-tail knockout already isolates the relevant
captured boundary with registered AIter all-reduce:

| profile | rank-max time |
|---|---:|
| `wo_b + AR` | 55.640 us |
| AR only | 26.581 us |
| empty graph | 1.100 us |

Therefore the direct component bounds are `wo_b=29.059 us` and
`AR=25.481 us`. The full inverse-RoPE + `wo_a` + `wo_b` + AR tail is
132.107 us.

There is also an exact-purpose local row-chunk oracle for the producer chain
`wo_a -> layout restore -> wo_b`:

| producer schedule | time | overhead vs serial |
|---|---:|---:|
| full M32 serial | 73.99 us | -- |
| 2 row chunks | 94.10 us | +20.11 us / +27.2% |
| 4 row chunks | 141.45 us | +67.46 us / +91.2% |
| 8 row chunks | 271.81 us | +197.82 us / +267.4% |

The chunked outputs also differ by up to 0.0078125 because the smaller-M GEMM
choices change reduction association. Thus the existing row-chunk producer is
both slower and not bitwise exact.

## Ideal overlap upper bound

Ignoring every chunk launch, publication and barrier cost, equal chunks in a
two-stage `wo_b -> AR` pipeline take

```text
T(K) = max(W,A) + min(W,A)/K + graph_floor
```

with `W=29.059 us`, `A=25.481 us`, and `graph_floor=1.100 us`:

| chunks | impossible ideal time | saving vs 55.640 us |
|---:|---:|---:|
| 2 | 42.900 us | 12.740 us / 22.9% |
| 4 | 36.529 us | 19.111 us / 34.3% |
| 8 | 33.344 us | 22.296 us / 40.1% |
| infinity | 30.159 us | 25.481 us / 45.8% |

Even the infinite-chunk oracle can remove at most one AR component. Across 43
layers that is 1.096 ms/model-step. At the contemporary TP4 BS32 resident rate
near 615 tok/s, one M32 step is about 52.0 ms, so the physically impossible
upper bound is only about 2.1% end-to-end throughput. K=2 and K=4 upper bounds
are approximately 1.1% and 1.6% respectively.

The measured producer penalty already exceeds the overlap opportunity. Even
granting proportional, zero-fixed-cost AR chunks, combining the measured
two-chunk producer (`94.10 us`) with the ideal two-stage fill/drain term
(`25.481/2=12.741 us`) gives about 106.84 us, slower than serial producer plus
whole AR (`73.99+25.481=99.47 us`). Four and eight chunks are much worse.

## Why the existing AR cannot simply be chunked

- AIter registered AR is one whole-buffer collective with fixed entry/exit
  barriers; K row chunks require K rendezvous and do not scale as `A/K`.
- Production's TP4 two-stage BF16 sum uses row-dependent owner rotation in
  8-row bands. A new chunk primitive must preserve that exact rank order.
- The prior direct tile publication/reduce protocol already solved IPC offsets
  and passed 1000 replay exactness, but measured 83.107 us versus 65.480 us for
  the same producer plus stock AR (21.2% slower).
- The AR+MHC structural oracle likewise found AR-only around 26.2--26.4 us and
  rejected finer peer-read fusion because repeated barriers dominated.

## Decision

Do not extend the current layer-20 oracle into a row-chunk implementation. The
serial producer overhead is already larger than the K=2 overlap budget, the
current chunk spelling is not bitwise exact, and the absolute infinite-pipeline
ceiling is below 3% end-to-end. Revisit only if a single persistent kernel can
produce rows and perform the existing owner-ordered reduction without per-chunk
collective entry/exit barriers; that is a new fused collective, not an
extension of the current registered AR call.

