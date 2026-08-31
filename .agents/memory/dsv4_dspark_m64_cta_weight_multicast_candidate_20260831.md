# DSV4 DSpark M64 CTA weight-multicast candidate review (2026-08-31)

## Decision

The only reviewed M64/TARGET_VERIFY candidate with a credible standalone
budget above 50 us/layer is an **A4-preserving, CTA-level weight multicast**
for routed gate/up and down.  It is not approved for production.  The next
step, if pursued, is a single-GCD recorded-route oracle only.

The candidate is strictly guarded by all of:

```text
forward mode = TARGET_VERIFY
DSpark gamma one target tier = M64
TP4 / EP1 / no A2A
E256 / top-k 6 / H4096 / local I512
original packed FP4 weights and E8M0 group-32 scales
learned-router layers only
```

Hash-router layers and every non-M64 tier remain on the current path.  No
weight conversion, approximate routing, altered reduction order, or host
occupancy synchronization is allowed.

## Why attention is not the next 50-us candidate

The relevant M64/DSpark attention records are already restrictive:

- The accepted C128 CK/MFMA sparse core changed roughly `264.48 -> 155.73 us`
  standalone, but service gained only about 1.0--1.15% because the attention
  branch is already overlapped.
- Sharing the nearly identical anchor/draft KV gather stream was bitwise exact
  for 100 mutations and 1,000 graph replays, but changed only
  `30.995253 -> 29.575506 us`, saving 1.419747 us/layer.
- M64 attention-tail decomposition measured `wo_b` around 30.59 us and the
  registered all-reduce around 42.81 us.  The impossible ideal overlap was only
  about 21.4 us, while two M32 producer chunks already added 26.57 us.
- Draft local-Q-head removal is useful, but it optimizes the draft rather than
  the dominant M64 target body.

The gamma-one observer instead places routed FP4 at about 0.65--0.70 ms of the
1.43--1.45 ms/layer rank-max target path.  That is the remaining component with
a credible 50-us structural budget.

## Evidence and theoretical budget

Across the recorded full-M64 target passes, learned-router medians/means are
approximately:

```text
active experts:          107.38
current A4 scans:        150.75
assignments in runs >4:  61.4%
maximum occupancy:       33.23 mean
```

The current A4 kernel therefore rereads/redecodes a weight row for roughly
`150.75 - 107.38 = 43.37` additional A4 chunks.  A multicast CTA that handles
up to four A4 chunks in parallel can approach one physical weight-tile load
per expert per four chunks, without increasing the per-wave assignment count.
The source-level scan reduction ceiling is about 28.8% on the learned layers.

Each TP4 expert shard contains about 3 MiB of packed gate/up/down weights.  The
corresponding source-request reduction ceiling is about 130 MiB/layer.  At the
measured roughly 1.0 TB/s packed-reader rate that is about 130 us.  Existing
L1/L2 reuse means this is not an HBM saving guarantee, but multicast also
removes duplicate VMEM issue and duplicate E2M1 decode.  Equivalently, applying
the 28.8% scan ceiling to the roughly 0.61--0.65 ms scan-dominated gate/down
body gives a 175--187 us optimistic ceiling.  Realizing only 28--30% of that
ceiling is enough to save 50 us/layer.

A 50-us/layer target-body saving removes about 2.15 ms from a 43-layer target
pass.  Against the observed 63.65-ms M64 target interval it is approximately a
3.4% target improvement.  This is meaningful, though nowhere near sufficient
alone for the final throughput goal.

## Concrete data flow

The sorter continues to produce stable expert-contiguous A4 chunks.  A
graph-safe device metadata pass groups consecutive chunks into descriptors:

```text
descriptor = {
    expert_id,
    chunk_id[1..4],
    chunk_count,
    stable partial-slot destinations
}
```

There is no D2H count or host-selected occupancy mode.  For the first oracle,
the real recorder may provide descriptors offline; device descriptor building
is a separate continuation gate.

For a hot descriptor, a four-wave pod executes:

```text
one wave  <-> one ordinary A4 chunk (four assignments)
four waves run their chunks in parallel
                    |
                    v
cooperatively load one packed K tile for the common expert/output-row tile
                    |
cooperatively decode E2M1 once into a small padded LDS tile
                    |
all four waves read that tile and execute the unchanged SDOT4 A4 math
                    |
unchanged DPP/subgroup reduction and unchanged stable partial slots
```

Gate/up uses the existing A4/R2 accumulator and DPP tree in every consumer
wave.  Down uses the existing A4 logical-W2-scale math, router multiplication,
FP32 partial slot, and fixed final reduction.  Only the source of the exact
signed E2M1 byte values changes from repeated global-load/decode to one
CTA-shared tile.

Use K128 or K256 ping-pong tiles, not a full expert row.  The CDNA2 ISA supports
buffer-memory loads directly to LDS, `DS_READ_B128`/`DS_WRITE_B128`, and
work-group synchronization with `S_BARRIER`.  MI200 has 64 KiB LDS/CU arranged
as 32 banks.  A padded/swizzled double buffer should remain below 8 KiB/CTA.
All waves must execute identical barrier counts.  The first version may use
ordinary cooperative loads/stores; direct MUBUF-to-LDS is only a follow-up
after exactness and resource metadata pass.

Cold singleton chunks may initially use the unchanged kernel in the oracle.
A production-capable version would need either a single descriptor launch with
independent cold pods and multicast hot pods, or proof that the extra sequential
cold/hot launch costs less than the saving.  Concurrent cold/hot streams are
excluded because earlier disjoint-weight stream experiments increased HBM/L2
contention.

## Why this is not a repeated rejected design

- **Not A8:** each wave still owns only A4 and retains the existing eight
  gate/up accumulators.  Four A4 chunks execute on four waves concurrently;
  no wave receives eight assignments or a longer A8 dependency chain.
- **Not expert persistence:** no first wave serially traverses all chunks of an
  expert.  Chunk-level latency hiding is retained.
- **Not the hot-expert LDS rejection:** that oracle decoded a long row into
  wave-private LDS and made one wave traverse the expert run, producing a long
  tail.  This proposal stages only a small K tile shared by parallel consumer
  waves.
- **Not permanent prepack/int5:** checkpoint bytes remain packed FP4 in HBM.
  Expansion exists only for the current tile in LDS, so there is no 1.25x/2x
  resident-weight or HBM-byte penalty.
- **Not MFMA:** the current exact `V_DOT4_I32_I8`/SDOT4 arithmetic and reduction
  order are retained.  The already-slower FP4 MFMA/UDOT alternatives are not
  revisited.
- **Not multi-stream buckets:** the candidate reduces actual weight requests
  for hot chunks.  It does not merely split the same bytes among competing
  launches.
- **Not the CTA-local down consumer:** intermediate quantization is initially
  unchanged.  The candidate shares weights across A4 chunks; it does not
  redundantly quantize the same A4 activation in every output-row CTA.

## Oracle implementation gates

Implement only if a fresh analysis over all real gamma-one M64 passes confirms:

1. learned-layer multicast descriptors reduce physical packed-weight tile
   loads by at least 20% median;
2. at least 40% of assignments are covered by descriptors with two or more A4
   chunks; and
3. hash layers are excluded rather than forced through a weak-reuse geometry.

The isolated GCD-4 oracle must then satisfy all of:

- Existing gate/up BF16 intermediate, intermediate INT8 values/scales, down
  FP32 partials, and final BF16 output are bitwise exact.
- 100 independent activation/router/route/weight mutations are bitwise exact.
- 1,000 HIP Graph mutation replays are bitwise stable.
- Seven-round symmetric ABBA on at least a median learned route, a concentrated
  route, and a dispersed route.
- No scratch or spills; LDS no more than 8 KiB/CTA; compiler and profiler show
  at least eight resident waves/CU for the hot kernel.
- Combined full routed stage saves at least **50 us/layer** on the median real
  M64 learned route.  Neither gate nor down may regress more than 5% in its hot
  substage, and excluded hash layers must remain byte-for-byte unchanged.
- Packed-weight VMEM requests or measured L2 bytes for covered hot descriptors
  fall by at least 20%; a timing win without the intended traffic proof is not
  enough to justify integration.

Stop immediately if barriers/LDS reads serialize the four consumer waves, if
VGPR occupancy drops below the declared gate, if the combined saving is below
50 us, or if the cold/hot join gives back more than one third of the isolated
hot saving.  Do not add a production selector or run a four-GCD AR/service test
until every single-GCD gate above passes.

## ISA references

The local MI200 ISA copy used for this review is:

```text
/home/pc/Code/Code/DOCs/instinct-mi200-cdna2-instruction-set-architecture.pdf
```

Relevant sections are 2.2.1 (64-KiB, 32-bank LDS), 9.1.9 (memory-buffer load
directly to LDS), 12.5 (`S_BARRIER` and `S_WAITCNT`), and 12.12
(`DS_WRITE_B128`/`DS_READ_B128`).

## Oracle-only implementation checkpoint

The first implementation is deliberately unreachable from production:

```text
python/sglang/kernels/jit/csrc/deepseek_v4/
  gfx90a_fp4_cta_weight_multicast_oracle.cuh
python/sglang/kernels/ops/moe/
  gfx90a_fp4_cta_weight_multicast_oracle.py
scripts/rocm/
  bench_dsv4_dspark_gamma1_m64_cta_weight_multicast.py
scripts/rocm/csrc/
  dsv4_m64_cta_weight_multicast_resource.hip
```

The gate core uses four K1024 phases.  Lanes 0--31 retain groups `0..31` then
`64..95`; lanes 32--63 retain groups `32..63` then `96..127`.  Thus every lane
visits exactly the same two K groups and in the same order as the accepted
wave64 kernel.  The down core retains one subgroup16 per R2 output tile and
maps the four waves only across consecutive A4 chunks.  Original encoded
`(slot << 24) | token` values are never renumbered, so candidate partial writes
land in the unchanged `[M,T,N]` slots.

ROCm 7.14 `hipcc --offload-arch=gfx90a -O3` parsing and explicit code-object
instantiation passed without touching a GPU.  Final metadata after reducing
the gate tile from K2048 to K1024 is:

| core | VGPR | SGPR | LDS | scratch | spills | workgroup |
|---|---:|---:|---:|---:|---:|---:|
| gate/up | 108 | 100 | 5,248 B | 0 | 0 | 256 |
| down | 64 | 85 | 5,248 B | 0 | 0 | 256 |

The gate initially compiled at VGPR103/LDS9,472 B; it was not retained because
it exceeded the declared 8-KiB LDS gate.  K1024 adds two work-group barriers
per K half but brings both kernels under the LDS limit.  Runtime occupancy is
still unproven and remains a mandatory stop gate.

CPU-only descriptor auditing on the real record 117 / forward pass 64 / layer
20 produced:

```text
active experts:       105
A4 blocks:            149
hot blocks:            69
cold singleton blocks: 80
multicast descriptors: 29
physical weight loads: 109
load reduction:        26.8456%
```

The descriptor builder proved that all 149 blocks are partitioned once and
that all 384 valid `(token,slot)` pairs are unique and covered.  This passes
the predeclared 20% load-reduction gate.  The runnable harness composes the
accepted cold kernel and multicast hot kernel sequentially, then requires
bitwise equality at every intermediate and a 50-us complete-stage saving.

GPU execution remains deferred.  At the checkpoint, GCD 4 was occupied by an
external BIO process and GCDs 0--3 by another service.  No process was killed,
and the oracle was not moved to another GCD.
