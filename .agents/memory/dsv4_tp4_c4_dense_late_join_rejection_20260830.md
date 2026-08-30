# DSV4 TP4 C4 dense direct-slot / late-join rejection (2026-08-30)

## Question

For the dense C4 regime (`c4_seq_len <= index_topk == 512`), test whether the
current

```text
core producer + index producer
-> wait both
-> zeros[M,512]
-> production topk_transform
-> attention consumer
```

can profitably become

```text
core producer + index producer
-> wait core only
-> directly emit sequential physical C4 slots
-> attention consumer
-> late wait for index producer
```

The continue gate was a gross saving of at least 30 us per C4 layer.

## Oracle scope

- GPU: physical gfx90a GPU 6 only.
- M=64, C4 page size=64, Top-K=512.
- Required edge lengths: 0, 1, 127, 128, 511, 512, plus ragged random rows.
- Non-contiguous, per-request randomized physical page tables.
- Reference transform: SGLang `topk_v1.cuh`, built through its JIT wrapper. The
  standalone DS environment did not have the monolithic AOT `sgl_kernel` op
  loaded, but this is the same production topk-v1 source and dense fast path.
- Direct transform: one full-wave Triton/HIP kernel emitting
  `page_table[row, logical//64] * 64 + logical%64`, padded with `-1`.
- Representative index producer: the exact C4 index-compressor WKV projection
  shape, BF16 `[64,4096] x [512,4096]`, followed by a deterministic future-state
  write. This was **not** a complete runtime compressor dump/callable, so the
  late-join result is explicitly a schedule simulation rather than service
  proof.
- Representative attention-consumer window: slot checksum followed by BF16
  `[64,1024] x [4096,1024]` projection.
- Every path was captured as a HIP graph with graph-safe events and persistent
  output addresses.

## Correctness

```text
100 input/page-table mutations: pass
1000 candidate graph replays: bitwise stable
candidate physical slots vs explicit reference: exact
reference vs candidate physical-slot sets: exact
index future buffer A/B: bitwise exact
production-order permutations: 0 / 6400 rows
mean permuted positions: 0.000
```

The zero-score tie concern does not apply here. `topk_v1.cuh` detects
`seq_len <= topk` and calls `naive_transform`, which emits the same ascending
logical indices as the proposed direct kernel. There is no attention reduction
order change in this dense regime.

## Seven-round graph timing

All numbers are trimmed means in microseconds (drop min/max):

| Profile | Time (us) |
|---|---:|
| A: current double-join + zero/topk + consumer | 36.977 |
| E: early-join + direct slots + consumer | 35.094 |
| B: late-join + direct slots + consumer | 35.205 |
| current zero/topk transform only | 6.608 |
| direct-slot kernel only | 5.716 |
| representative consumer only | 33.283 |

Derived:

```text
gross A -> B saving:              1.772 us (5.03%)
direct-slot isolated saving:      0.892 us
late-join simulated E -> B:      -0.110 us
continue gate:                    >=30 us
result:                           reject
```

## Interpretation

The current production topk kernel has already reduced the dense case to a
direct sequential page-table transform. A new specialized kernel can remove
the score zeroing and some generic launch overhead, but the entire isolated
budget is below 1 us. In this workload the index producer is already hidden by
the core/consumer critical path; moving its join later produces no measurable
benefit.

Because the producer was an exact-shape projection plus representative post
work rather than a full runtime replay, this oracle alone could not prove a
positive late-join result. It is nevertheless sufficient to reject production
integration: the measured gross saving is only 1.8 us, the isolated direct-slot
op saves under 1 us, and the late-join component is neutral. This is far below
the predeclared 30-us gate and cannot materially advance the 1.5k tok/s target.

The standalone implementation was removed after the rejection; only this
experiment record is retained.

