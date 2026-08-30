# TP4 M32 wq_b Q-epilogue lower-bound rejection (2026-08-30)

## Question and oracle

The candidate was to fuse per-head Q RMSNorm/RoPE into the `wq_b` GEMM
epilogue, avoiding the 512-KiB raw BF16 Q write/read before
`fused_qk_norm_rope_swa_store`.  The current consumer also owns KV
norm/RoPE/cache-store, so removing the entire consumer was not a valid bound.

An oracle-only `HAS_Q=false` specialization was added to the existing Triton
kernel. Production calls retain the original Q+KV grid. The standalone oracle
uses the real layer20 M32 positions and rank0 Q tensor, tiled from the TP8 H8
dump to the exact TP4 H16/D512 shape. It compares:

- A: current Q+KV consumer;
- B: a pre-generated exact `q_out` (free hypothetical GEMM epilogue) plus the
  same kernel executing only KV norm/RoPE/BF16 unified-cache store.

Both arms copy the same real-shaped KV source before the consumer. The free Q
result is prepared outside B's timed/captured body.

## Correctness and result

- 100 bounded Q/KV mutations: `q_out`, mutated KV and the 32x512 cache bytes
  were bitwise exact; the KV-only kernel did not modify pre-generated `q_out`.
- 1000 captured graph replays retained the same exact outputs with no stale
  cache bytes.
- Seven-round ABBA measured current Q+KV `9.029 us` and KV-only `8.325 us`.
  Even deleting all Q consumer work for free saves only `0.704 us`, far below
  the required `5 us` continuation gate.

The 16 Q programs execute concurrently with the single KV program; kernel
launch and the KV/cache tail dominate elapsed time. Raw-Q traffic volume is not
on the exposed critical tail. Therefore a custom CK/HIP `wq_b` GEMM epilogue
cannot recover the hoped-for 5+ us even under an impossible free-Q assumption.
Do not implement it.

Standalone artifacts only:

- `scripts/rocm/bench_dsv4_tp4_wqb_q_epilogue_lower_bound.py`;
- default-preserving oracle constexpr in
  `fused_qk_norm_rope_store.py`.

No production selector was added and no commit was made.
