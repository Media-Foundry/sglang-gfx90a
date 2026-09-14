# Read-only attention audit while comb-refinement ABBA runs

Motivation: current mix8 C16x8K profile has2.771669s/wave in main sparse
attention. No new GPU test or production change was made in this audit.

## Actual HIP path, not the similarly named CUDA implementation

`DeepseekV4HipRadixBackend._forward_unified_kv` calls
`unified_kv_kernels.runtime.build_prefill_indices` then `runtime.prefill`,
which reaches `paged_prefill.sparse_attn_v4_paged_prefill` and the gfx90a
Triton fallback. The CUDA backend's `_forward_prefill_sparse`/FlashMLA and
its combined-workspace ABI are not the path to optimize for this profile.

The current fallback already uses one wave, BLOCK_H16/BLOCK_D512/BLOCK_K16.
History already measured8/4/2/1 waves and selected1 on TP4; do not count that
gain again. Model code's unified-KV branch already allocates only local heads
(TP8:8) and bypasses legacy64-head zero padding. The M16 arithmetic tile
still has8 masked rows, which is different from a64-head tensor allocation.

## Do not port indexer empty-tile skip blindly to main attention

The attention kernel masks invalid KV loads but still performs dot operations
for such a tile. However, its caller already constructs ragged lengths:
`_prefill_lengths_kernel` counts `pi>=0`; the builder copies only that valid
front-packed compressed prefix plus actual prior-window slots. Extend indices
cover only the actual causal window. Normal valid/front-packed input therefore
has no whole padded512-key tail. At most the last16-key tile is partially
masked; a fully empty region executes no loop iterations.

This source-level conclusion assumes the documented front-packed metadata
contract. It does not prove arbitrary malformed/holey inputs are safe, or
replace a runtime metadata oracle. It is enough to reject a speculative
"main attention empty-tail skip" speed project without first showing real
empty blocks. The earlier indexer rectangular-grid waste was different.

## Cached code-object inventory (not live-module attribution)

The server redirects Triton cache through SGLANG_CACHE_DIR to
`/home/pc/.cache/sglang/triton`, not the standalone default `/home/pc/.triton/cache`.
The first default-cache filename search returned no matching kernel; no missing
module or failed compilation was inferred from that empty search.

Two cached gfx90a H8/D512 one-wave artifacts both show:

- launch shared memory16384 bytes; static LDS0;
- `amdhsa_next_free_vgpr=354`, accumulator offset256, AGPR count98;
- private scratch0.

Do not report354 ordinary VGPRs: the directive includes accumulator mapping.
Do not derive exact occupancy from this inventory alone. Other H16 cached
experiments do have scratch, but those are not evidence of an H8 spill problem.
Neither cached artifact is proven to be the service's currently loaded module;
a future isolated invocation must record its actual compiled instance.

Reproducible read-only script and hashed inventory:
`.agents/experiments/dsv4_tp8_prefill_attention_audit_20260915/audit.py`
and `cache-inventory.json`. No speed claim follows from this audit.

Next reasonable screen, after current service ABBA releases GPUs: compare the
current full paged-prefill operator with any reused CK-style candidate at
H8 and actual two-source/ragged shapes, counting index preparation and complete
attention output. Preserve per-query indices, prefix/extend order, duplicate
occurrences and sink semantics. Do not begin another implementation by deleting
guards or by treating a decode microbenchmark as proof of large-M performance.
