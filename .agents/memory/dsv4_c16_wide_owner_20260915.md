# Wide query-owner eight-rank synthetic screen and result recheck

2026-09-15, starting HEAD ae99078894. Original V4 prefill work only.
No production source or launcher defaults changed. No new model-service speed.

## Rechecked earlier evidence

Re-executed all assertions from the immutable prewarm-service and long-attention
analyzers without their output writers, and compared recomputed objects with
saved analysis.json. Both matched exactly. This rechecks raw timestamps,
input echoes/cache counts, rank coverage and compiled artifact matching.

- C16 x16K warm: 6978.102488 input tok/s.
- C16 x32K warm: 5790.835165 input tok/s.
- Same-live-input stages1/default attention comparisons: 16512 byte-exact calls,
  all eight ranks and43 target layers, actual M32767/32768.

These do not establish global determinism. Accepted default8K service remains
8384.698465; optional non-bitwise producer stack remains8692.442097.

## New bounded screen

Eight ranks, original query16 runtime-M score kernel, existing deterministic
Top-K, existing host_plan and reconstruct. The experiment directly invokes
runtime-M on small owned batches; it does NOT remove either production guard
(owner width<=2048; wide wrapper full-M>=8192).

Synthetic FP8 Q and packed/preshuffled index cache, variable FP32 scales,
BF16 dot. This is NOT a live model fixture. Every rank hashes all replicated
Q/weights/lengths/pages/cache before rank-specific physical page permutation;
all eight hashes match per case. Each rank then renumbers physical pages by
a different permutation. Only logical Top-K IDs cross RCCL; reconstruction
uses local physical pages. Full Q producer and compressor remain unchanged
and are outside BOTH timing paths.

Candidate timing includes GPU index_select packing, scores, Top-K, int32 RCCL
all_gather_into_tensor, and full-row reconstruction. CPU host planning is
offline and excluded. Baseline is full replicated query16 scores+Top-K, not
the slower historical per-row implementation. Three ABBA cycles, three calls
per sample. For each sample take the slowest rank, then median across six
samples per arm; do not sum independent component medians.

| Fixture | M / C4 width | Owned rows/rank | Baseline ms | Owner ms | Component ratio |
|---|---|---:|---:|---:|---:|
|Two16K causal requests|32768 /4096|3584|38.956417|5.593545|6.9645x|
|One32K ragged request, prefix1|32767 /8192|3840|80.026370|10.411354|7.6865x|
|Four irregular mixed-prefix requests|32768 /8192|3840|75.520376|10.061341|7.5060x|

All ranks passed full score-byte and logical/physical-ID comparisons for three
inputs per case: original, head permutation+weight mutation, and all-zero Q
cutoff ties. Cross-rank full logical IDs also compared. Timing uses regenerated
non-tie Q, checked before timing. Noncontiguous full page-table stride is NP+3;
owner packing produces contiguous pages. Production kernel source hashes
were frozen and rechecked at completion.

This is encouraging enough for a later live-input oracle and guarded service
ABBA. It does not prove6.9–7.7x model speed, real rank KV equality, all shape
coverage, graph safety, peak service VRAM, or1M-pool capacity after integration.
No new service has been started in this screen.

## Harness failure retained in account

Initial torchrun used only PYTHONPATH=python and failed before running the
oracle while importing sgl_kernel. The DS launcher also adds its two local
AOT Python paths. Correct run.sh reproduces those paths and SGLANG_USE_AITER=1;
then all eight workers completed normally. This was launch environment setup,
not permission denial, GPU failure, or a failed numerical test. The initial
tool traceback is in the conversation; run.log is the successful second run.
AIter emitted existing NUMA-balancing warnings; no system setting was changed.

## Reviewer follow-up after this cycle

Review f2c8524d predates accepted stages1 and optional producer measurements.
Do not repeat owner producer as untested or budget the old2.76s attention as
entirely unoptimized. Wide-owner work addresses16K/32K coverage, not the already
small8K owner score chain. For the8K main line, next priority is updated profile
and a V4-compatible post-to-projection-partial MHC oracle, preserving current
BF16 residual rounding, FP32 Fn, K1024 reduction contract and current-layer
coefficients. H16 activation exchange should first clear single-GCD compute
and real paired-Q/KV equivalence gates; paper/backend claims remain unverified
in this cycle. No V4.1 coefficient-lag semantics are authorized.

Artifacts: .agents/experiments/dsv4_c16_wide_owner_20260915/{oracle.py,run.sh,
recheck.py,result.json,run.log}. No large tensor dumps or new persistent cache.
