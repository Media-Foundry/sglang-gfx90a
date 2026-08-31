# DSpark auxiliary hc_post reuse checkpoint (2026-09-01)

## Change

DeepSeek-V4 DSpark captures completed hidden states from target layers
40/41/42. Under cross-layer MHC fusion the old loop called `layer.hc_post()`
again for every captured layer, even though the next layer's fused boundary
had already produced that exact completed residual. The final target layer was
also recomputed once for capture and again for the model output.

The checkpoint now optionally returns the already-completed previous residual
from a decoder layer only when the DSpark model loop asks for it. Layer 40 is
tapped at the layer-41 boundary, layer 41 at the layer-42 boundary, and the
final completed output is reused for layer 42. Non-fused execution retains the
old self-contained behavior. Native AR does not request DSpark auxiliary
states and never enters the extra return path.

This is an exact common-subexpression elimination, not an approximate draft
row skip: `apply_mhc_post_pre_boundary()` defines the returned residual as the
completed previous-layer `hc_post`, and its fallback explicitly runs the same
`hc_post` before assigning that residual.

## Validation

- Original weights, TP4/EP1, physical GCDs 4,5,6,7.
- Static gamma-three DSpark, BS32, 32 distinct concrete coding requests.
- France first nine token IDs exact and semantic Paris in all three rounds.
- Every request generated 256 tokens with `finish=length`.

Resident BS32 throughput:

```text
930.140, 903.178, 883.113 tok/s
median 903.178 tok/s
```

The prior same-profile control center was about 901.6 tok/s. The measured
difference is only about +0.17% and is noise-scale; do not advertise it as a
performance checkpoint. Keep the change because it removes three redundant
M128 MHC completions per target step without changing model mathematics.

Raw report: `/tmp/dsv4_gamma3_aux_hcpost.json`.
