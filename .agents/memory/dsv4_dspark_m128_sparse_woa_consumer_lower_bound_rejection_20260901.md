# DSpark M128 sparse-attention / wo_a consumer lower-bound rejection (2026-09-01)

## Goal

Screen the last attention-output fusion proposal with a strict continuation
gate: replace

```text
CK split2 sparse core -> FP32 reduce -> BF16 output -> inverse RoPE
-> BF16 output -> grouped wo_a
```

with a four-wave M16N64 consumer that reduces the split workspace, preserves
both BF16 rounding boundaries, applies inverse RoPE while staging A, and feeds
M16N16K16 BF16 MFMA directly.

The proposed production guard was DSpark TARGET_VERIFY, TP4, BS32, M128,
H16/D512 only; native AR was never modified.

## Physical-GPU screen

- `amd-smi process --general --sort-by-pid`: no running GPU processes.
- Physical GPU: 4.
- Shape: M128/H16/D512, ragged Top-K/context 512, grouped wo_a
  `[4,512,2048]`.
- Existing complete baseline (CK core + production inverse RoPE + einsum):
  **214.906 us/layer**.
- First fused MFMA layout: **276.471 us/layer**, a **61.565 us regression**.
- The first layout was graph-stable but did not yet reproduce the reference
  output (`max_abs=0.328125`, `relative_l2=1.0`), so it was removed rather
  than retained as production-quality code.

## Lower-bound decision

The decisive result is not the incorrect candidate output; it is the resource
lower bound. The already-correct standalone CK sparse core alone is about
182 us at context 512. The 5% E2E continuation gate required the complete
fused chain to save at least 85 us/layer, which would require a total below
about 130 us/layer. That is lower than the unchanged sparse core itself.

Therefore this fusion cannot clear the checkpoint gate without also replacing
the sparse core, and the current service trace reports a substantially smaller
exposed attention-core span than the standalone oracle. Stop this direction.
Move the >=5% search to the 579--619 us/layer MoE rank-max region, especially
the 445--475 us routed branch and its 79--91 us TP4 collective.
