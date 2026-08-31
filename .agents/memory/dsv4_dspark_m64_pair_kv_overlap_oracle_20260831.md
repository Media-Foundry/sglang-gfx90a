# DSV4 DSpark gamma-one M64 pair-KV overlap oracle (CPU-only scaffold)

## Question

Gamma-one target verification lays out each request as adjacent
`[anchor, draft]` rows.  The anchor cannot be removed: it is the previous
step's frontier bonus token and has never been forwarded by the target model.
The remaining exact reuse opportunity is therefore inside the existing M64
attention call: load common prefix KV once for both adjacent queries while
retaining independent causal masks and softmax states.

## Offline analyzer

`scripts/rocm/analyze_dsv4_m64_pair_kv_overlap.py` consumes the production CK
replay payload produced by
`python/sglang/kernels/ops/debug/dsv4_ck_replay.py`.  It is CPU-only and does
not import SGLang or initialize HIP.

For each adjacent pair it reports:

- set intersection, Jaccard, and overlap relative to the shorter row;
- exact sequence/set equality and anchor/draft prefix relations;
- common ordered-prefix length;
- the best small displacement between gather streams, matching entries, and
  longest equal run;
- reusable full 8/16/32-entry tiles at that displacement;
- physically consecutive shared-slot tiles after mapping compact dump indices
  back through `physical_slots`.

The aggregate set-reuse bound counts shared slots once instead of twice.  It
is deliberately labelled an upper bound: unordered intersection does not
prove a pair-query kernel can reuse a tile without extra gather/control cost.
The best-shift bound is more restrictive but still excludes added VGPR and
occupancy costs.

## Production result

A fresh eager capture used physical GCDs 4--7 and 32 distinct coding prompts.
The strict payload is:

```text
/tmp/dsv4_pair_dump/layer_3_rank_0_c128_unified_sparse_m64.pt
```

Analyzer result over all 32 adjacent pairs:

```text
Jaccard mean / median:                  0.959744 / 0.960769
overlap relative to shorter row:       1.000000
ordered common-prefix ratio:           1.000000
anchor-is-prefix fraction:             1.000000
position-adjacent fraction:            1.000000
set-load reduction upper bound:        0.489809
best-shift load reduction upper bound: 0.489809
best shift:                            0 for all 32 pairs
longest equal run mean / median:        24.03 / 24.50 entries
shared aligned tiles:                  82 x K8, 32 x K16
```

Thus every anchor gather stream is the complete ordered prefix of its paired
draft stream; the draft adds only its newest causal entry.  This is strong
enough to justify an oracle-only pair-query CK kernel.  It does not prove an
end-to-end win because adjacent independent CTAs may already reuse the same KV
through L2, while explicit pairing doubles softmax/PV state and VGPR pressure.
Require at least 15 us/layer standalone before production integration.

The analyzer's synthetic shifted-window and exact-pair tests passed, as did
`py_compile` and `git diff --check`.  Reproduction:

```bash
python scripts/rocm/analyze_dsv4_m64_pair_kv_overlap.py \
  /tmp/dsv4_pair_dump/layer_3_rank_0_c128_unified_sparse_m64.pt \
  --json /tmp/dsv4_pair_overlap_layer3.json
```

The analyzer defaults to the strict C128/M64 contract (`64` rows,
`compress_ratio=128`) and fails rather than silently mixing C4 or another graph
tier.

## Pair-query CK static implementation

The real row lengths are especially favorable: anchors contain 21--29 indices
and drafts contain 22--30, always with exactly one appended slot.  Every row
therefore occupies two 16-key tiles.  Production split-K=2 gives the same tile
boundary to both rows, so both the full first tile and the partial tail tile are
prefix-compatible.  Loading the longer draft tail once is exact: the anchor
keeps its original smaller `valid_keys` mask, making the extra draft key's
anchor probability zero before PV.

The oracle-only core is
`gfx90a_dsv4_unified_sparse_pair_oracle.cuh`.  One 512-thread CTA owns one
anchor/draft pair and one original split.  Two four-wave groups retain the
production QK K128 ownership, four-wave FP32 sum order, online-softmax tile
order, D128 PV ownership, split2 workspace, and unchanged final reducer.  A
tile aliases one LDS bank only when the shorter physical-slot vector is a
prefix of the longer; incompatible inputs use two banks.  No production
selector imports this header.

ROCm 7.14 `hipcc --offload-arch=gfx90a` syntax and object compilation passed.
Code-object metadata for the paired core is:

```text
VGPR 128, SGPR 60, LDS 42,500 B, scratch 0, spills 0, workgroup 512
```

The production qreg+KV-prefetch core compiled beside it is VGPR 166 / SGPR 39 /
LDS 21,248 B / workgroup 256.  Pairing halves the core grid from 128 to 64 CTAs
and its LDS permits only one CTA/CU, so the near-49% load bound must overcome
reduced cross-CU occupancy.  The continuation gate remains 15 us/layer.

The pending single-GPU harness is:

```bash
HIP_VISIBLE_DEVICES=4 python \
  scripts/rocm/bench_dsv4_dspark_m64_pair_ck_sparse.py \
  --replay /tmp/dsv4_pair_dump/layer_3_rank_0_c128_unified_sparse_m64.pt \
  --output /tmp/dsv4_dspark_m64_pair_ck_sparse.json
```

It requires 100 data mutations to remain bitwise equal to the existing CK
path, 1,000 graph replays to remain bitwise stable, and seven-round ABBA saving
of at least 15 us.  GPU execution was intentionally deferred after static
compilation pending explicit coordination.
