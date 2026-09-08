# TP8 no-A2A EP2/expert-TP4 component coverage — 2026-09-08

Parent87ece0fbcb. Production3325034 unchanged, nativeTP8/EP1,1M KV pool.
Previous no-A2A owner oracle tested TP4 E256/I512 versus E128/I1024;
Mori TP8/EP2 is another architecture. Neither directly covers today's
E256/I256 versus E128/I512 with accepted prefetched gate.

GPU4 standalone, AMD-SMI pre-audit no foreign owners. Use pass80 of current
32-real-code-request recorder1788860446.3842516, layers0/20/40. Divide counts
by8 and require192 assignments. Reconstructed unique-top6 IDs reproduce exact
expert occupancy, NOT original token/expert correlations. Owners are balanced
offline for each sampled layer/pass; not a realizable online repartition.

BaselineE256/I256 G832; two ownersE128/I512, gate/down G832 or1664. Packed
weight elements per owner equal baseline. Random nonconstant scales, independent
synthetic weights. Dense intermediate quant, masked partial clear, down and
fixed reduction included; sorter, initial quant, real collectives, shared/MHC
interference excluded. Candidate uses same prefetch implementation, not old gate.

100 mutation comparisons against generic gate per shape (five profiles/layer),
all exact;1000 full-chain replays per profile stable and finite. This validates
prefetched kernel geometry, NOT equality between layouts or real shard loading.

Repeat with full artifact capture, five symmetric-order cycles,100 replays/sample:

|Layer|Baseline us|Best-owner maximum us|Component throughput change|
|---|---:|---:|---:|
|0 hash|280.464|260.867|+7.51%|
|20 learned|220.941|219.882|+0.48%|
|40 learned|226.595|223.777|+1.26%|

The first run had the same pattern (hash~8%, learned neutral/slight gain), but
tool stdout truncated its owner metadata. Adjacent JSON is the complete repeat,
not repaired/fabricated first-run output. Best-owner maximum is a sequential
single-GPU cost proxy, not an eight-rank rank-max measurement.

No sampled layout clears10% component continuation gate. Do not wire a global
model layout change from these results. This does not prove every layout or
kernel impossible: compact owned-only quant and broader geometry untested.
Hash-only benefit applies to three layers at most; no43-layer extrapolation.
Real weight-shard oracle and service parity would be required before deployment.
Production weight precision, KV capacity and selectors unchanged.
