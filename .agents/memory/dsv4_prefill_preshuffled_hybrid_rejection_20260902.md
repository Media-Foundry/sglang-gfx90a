# DSV4 prefill preshuffled hybrid rejection (2026-09-02)

## Scope

This experiment attempted to let the gfx90a direct/MFMA routed-expert kernels consume AIter's preshuffled FP4 weight layout. The goal was to keep the fast direct path for small batches while using AIter for large prefill batches without retaining two full weight layouts.

## Synthetic oracle

- Random addressing probes for 4096 gate/up and down weight/scale queries matched AIter's shuffle exactly.
- At M=2048, the complete synthetic routed stage improved from 10.625 ms to 10.217 ms (3.84%).
- Intermediate and final outputs were bitwise exact in that synthetic test.
- At M=1, the preshuffled direct path regressed from 51.69 us to 66.34 us (28%).

## Service result

The production model failed semantic correctness when the hybrid was forced. Decode emitted repeated/nonsensical text even though the synthetic oracle passed. Native AR also regressed to roughly 51--52 tok/s after warmup (first request included JIT and was about 13 tok/s), while the previously validated direct small-M path was materially faster.

A per-request prefill chunk cap was also tested with a large global chunk. It did not solve the throughput gap: the observed C32 aggregate prefill remained around 3.0--3.1k input tok/s, far below the 10k target. The constructed C1 request was about 2.33k input tok/s, but was not the standard 4604-token baseline prompt and therefore is not accepted as a regression comparison.

## Interpretation

The synthetic `shuffle_weight_a16w4` oracle did not prove that the tensors exposed by the real model loader have the same gate/up projection mapping and transformation history. Correct individual address formulas are insufficient when the production loader may already merge, shard, or transform logical projections differently.

## Decision

- Reject and remove the production-adjacent `preshuffled` kernel selector and scheduler hybrid.
- Keep only the committed read-only preshuffle address probe as diagnostic infrastructure.
- Before revisiting this route, capture a real loaded layer's w13/w2 tensor and compare it against the corresponding raw checkpoint expert slice through every loader transformation.
- Never infer production correctness from synthetic shuffled tensors alone; require France semantic output plus fixed-token/teacher-forced checks after model integration.

