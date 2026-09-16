# Route-major CK stage2: exact, small isolated gain, not integrated

2026-09-16 after shorter-length regression and length profiles. Production
checkpoint unchanged:8K10205.686130 /16K10023.408672 /32K9605.065395 input tok/s.
OriginalV4/TP8/nativeAR/originalweights/1Mpool; no DSpark or V4.1 work.

Bounded experiment: reuse accepted independently rebuilt unique-Set CK binary
unchanged. GPU pack BF16 stage1 intermediate into sorted-assignment order;
identity virtual IDs produce route-major FP32 partials; inverse map reads the
original six slots in fixed order and rounds once toBF16. No expert pruning,
weight conversion or changed reduction order. No runtime selector changed.

Two checksum-verified historical real fixtures: M8192 andM32767,Top6/I256/H4096.
CPU route completeness checks require every(token,slot) exactly once. Five
numeric mutations including reversed expert-block ordering; all mappedFP32
partials and finalBF16 outputs byte-exact;100 graph replays exact. Fresh
breakdown run independently repeats the same correctness checks and ABBA.
HIP_VISIBLE_DEVICES5 resolves PCI0000:b3:00.0 (not rocm-smi card5). All TP8
services had stopped. Sessions5088 and21209 both exited0, GPUs released.

## Complete-stage event ABBA, three cycles per mode

| Fixture | Mode | Current remap+CK+vec4 ms | Pack+route CK+inverse reduce ms |
|---|---|---:|---:|
|M8192|eager|2.744049|2.672846|
|M8192|graph|2.741835|2.669486|
|M32767|eager|10.499910|10.189552|
|M32767|graph|10.494469|10.191579|

~2.59%-2.96% latency reduction, ~2.66%-3.05% throughput ratio. Second run
M32767 eager10.555219->10.188779,graph10.545966->10.204247. Preallocated
scratch in both arms; no stage1/sorter or service time included. Isolated
substage medians (diagnostic only; not independent additive service budgets):

| M32767 operation | Current ms | Candidate ms |
|---|---:|---:|
|remap vs pack+inverse|0.005152|0.229705|
|same unique CK instance|6.798173|6.340236|
|fixed-order reduction|3.743403|3.628551|

The additional pack consumes a substantial part of CK improvement. No claim
of lower total HBM traffic: same live partial write/read, plus pack traffic.
Scratch increases: M32767 partial3221127168->3489464320bytes, plus109045760
packedBF16 bytes (~360MiB additional excluding small maps). M8192 partial
805306368->1073643520bytes plus33551360packed bytes. No1Mpool capacity
validation for a service using this candidate yet, and no E2E gain claim.
Using a0.31ms stage2 saving x43layers x4forwards gives only~53ms conditional
8Kwave budget, about0.4%; not justification for claiming several percent E2E.

## Failure retained and corrected

Initial screen-v1 incorrectly required allocated sorter capacity divisible64.
Actual capacities65530 and212980 are not; valid padded lengths57152 and205056
are. First run exited1 before comparisons/timing. screen.json records failed,
screen.log and exact screen-v1.py preserve evidence; header unchanged. Corrected
screen.py checks valid%64==0 and valid<=capacity; outputscreen-v2.json is complete.
Do not report initial failure as CK illegal addressing or a hardware problem.

Artifacts .agents/experiments/dsv4_ck_route_major_20260916/:
route_major.cuh,screen.py,screen-v1.py,screen{,-v2}.{json,log},
breakdown.py,breakdown.{json,log}. No production integration/default promotion.

Next decision: this is a small candidate, not the main route to a large E2E
gain. A service ABBA would need retain1Mpool and justify extra scratch; first
consider whether producer output layout can avoid the extra pack WITHOUT
changing arithmetic, reintroducing atomics or creating cross-CTA spin protocols.
That is a new untested design, not an existing measured win. Another distinct
target is long-context indexer owner chain, now measured3.50s/32Kwave versus
0.290s/8Kwave. Goal remains active.

Read-only follow-up audit: gfx90a_indexer_owner.host_plan already assigns
original query16 groups with groups[r::8] (round-robin), not contiguous query
chunks per rank. Do NOT propose round-robin assignment as a new load-balancing
fix or blame contiguous ownership for the measured long-history cost. Further
indexer work first needs actual logits/TopK/pack/AllGather breakdown.
