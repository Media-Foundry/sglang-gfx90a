# H8 prefill geometry/lifetime screen and real order-sharing bound

Production unchanged: original V4 TP8/native AR C16 prefill7953.956103 input
tok/s, original weights and1M KV. All GPU work below uses physicalGCD4 only,
with no service running. These are not E2E speed results.

## Three rejected execution variants

Synthetic causal8K two-bank fixtures (C128, varied C4), H8/D512, M8192. Same
per-row selection, sink and K16 region partition. Three ABBA cycles, five calls
per event interval; common metadata construction/allocation excluded. Five
Q/K/sink scale mutations check full output bits/finiteness versus production.

| Variant | C128 control/candidate ms | C4 control/candidate ms | Output bits |
| --- | --- | --- | --- |
| Logical H8, one wave |2.000 /32.431|5.935 /96.341|0/5 exact, max_abs0.001953125|
| H16, two waves |1.985 /3.005|5.934 /9.969|5/5 exact|
| H16, one wave, volatile Q reload per KV tile |1.983 /3.512|5.927 /11.266|5/5 exact|

Production compiler resource report:354 registers,0 spills,16384B LDS.
H8 reports512 registers/1081 spills/32768B LDS; two-wave267/0/16384;
reload416/0/32768. These are compiler resource fields, not measured hardware
occupancy. The reload source did not achieve its intended register reduction.
All variants lose decisively; no M32768 or service benchmark warranted, and no
production selector is added. `screen.json` and `reload.json` retain samples,
checks, resource fields and HSACO hashes. They do not claim general exactness
or qualify changed precision for service use.

## Real C4 pair reuse: count first, no new kernel yet

`overlap.py` replays the production query16/runtime-M logits and deterministic
Top-512 on a SHA-verified saved layer20/rank0 fixture, M32767, first four requests
of a real heterogeneous C16 wave. Query/KV and input metadata are actual service
capture data. Three selection replays exact. Full logical IDs saved with hash.

Adjacent global query pairs keep original per-row K16 boundaries. Compare each
prefix tile's physical IDs at identical positions; shared tiles require no
conflicting valid IDs and independent row masks. Requests are not crossed.
Negative/sentinel positions do not authorize sharing conflicting valid slots.
Extend-window counts are inferred from the known zero-prefix flat SWA positions,
not from a saved extend-index tensor.

- Original prefix tiles921184; extend tiles260344.
- Compatible prefix tiles52855 (same result as literal equal-ID tiles).
- Compatible extend tiles1144.
- At most53999 of1181528 query-tile executions removed: **4.5703%**.
- Below512 keys:42080 shared tiles versus134560 original prefix tiles.
- Saturated512-key rows:10775 shared versus786592 original prefix tiles.

This is a work-count bound for one layer/forward, assuming free compatibility
testing, no extra Q/register cost and no occupancy loss. It is NOT a latency
bound or predicted whole-model improvement. It does not exclude a union schedule
that changes the original online-softmax grouping, nor different later layers.
Together with previous DSpark pairing failures, it does not justify immediate
development of a general order-preserving pair kernel for this workload.

Next candidate: indexer-only query producer ownership. The accepted owner path
still computes replicated full-M query projections (~0.627s/wave in current
profile), before distributing only scoring/selection. Check true inputs and
numeric contracts before attempting that; no such implementation is included here.
