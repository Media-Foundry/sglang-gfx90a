# DSV4 M32 sqrtsoftplus TopK + stable A4 fusion stop

Date: 2026-08-30

## Exactness audit

The active M32 path cannot import `aiter.topk_gating`; it falls back to
SGLang's generic `moe_fused_gate` small-token kernel. That kernel launches one
256-thread CTA per token, uses `log1pf(expf(x))`, selects six experts, and
renormalizes with `device::warp::reduce_sum<16>`.

The existing gfx90a HIP sqrtsoftplus router produces identical Top-6 IDs,
including bounded ties and adjacent FP32 values, but its weights differ by up
to `4.470348358154297e-08`. Changing its sequential six-value renormalization
to the same XOR `reduce_sum<16>` tree did not remove the difference. The
remaining discrepancy comes from separately compiled score/division math and
instruction contraction around exp/log/sqrt, not selection order.

The difference is not harmless under the project's exactness rule. Applying
the two weight sets to a synthetic real-shape `[32,6,4096]` BF16 expert output
and reducing in FP32 produced eight different BF16 output elements on the
first mutation, with maximum absolute difference `0.0001220703125`.

## Why a direct stable-sort fusion is structurally poor

A stable A4 sorter requires global counts, an expert prefix scan, and stable
rank in flattened token/slot order. The router naturally uses 32 independent
token CTAs. Those CTAs cannot safely construct final stable metadata in the
same ordinary kernel without a grid-wide barrier. Atomic expert cursors are
forbidden because arbitration changes same-expert assignment order.

A single 256-thread CTA could process all 32 routers and then sort, but it
would serialize 32 rounds of score evaluation and six dependent selections,
destroying token-level parallelism. A deterministic two-kernel design simply
recreates router output followed by the existing sorter; the standalone exact
static A4 sorter was already slower than AIter (15.527 versus 9.909 us in its
latest recheck).

## Decision

Stop the TopK+stable-A4 mega-kernel. It cannot satisfy both the required
bitwise/teacher exactness and the `>=8 us` chain-saving threshold with the
available decomposition. Revisit only if there is a cooperative-grid router
primitive with an explicit deterministic global barrier, or if correctness
policy permits the observed numerical drift.

The temporary production-header macro was removed. The independent diagnostic
script `scripts/rocm/bench_dsv4_m32_router_renorm_exact_oracle.py` is retained.
