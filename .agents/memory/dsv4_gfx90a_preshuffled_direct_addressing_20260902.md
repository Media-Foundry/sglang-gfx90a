# DSV4 gfx90a preshuffled direct addressing (2026-09-02)

## Motivation

The fastest small-M path consumes logical/raw FP4 rows, while AIter's best
large-M grouped path consumes the A16W4 preshuffled layout. Keeping both layouts
would duplicate roughly the entire routed-expert weight footprint and is not
viable. The intended solution is for the custom gfx90a kernels to address the
AIter layout directly, allowing one weight copy and shape-dependent dispatch.

## Layout formulas

Device helpers were added for gate/up and down packed-weight byte offsets. The
existing direct kernels already use the AIter-shuffled E8M0 scale offset formulas;
weight and scale layouts are different and must not share an offset calculation.

The probe samples logical `(expert, output_row, packed_k_byte)` coordinates from
raw tensors, transforms weights/scales with AIter's real
`shuffle_weight_a16w4`/`shuffle_scale_a16w4`, then reads the shuffled tensors on
the GPU through the new address helpers.

## Correctness

On physical GPU 4 (gfx90a):

```text
gate_up=True  queries=4096 weight_exact=True scale_exact=True
gate_up=False queries=4096 weight_exact=True scale_exact=True
```

The test uses independently randomized weight bytes, scale bytes, experts, rows,
and packed-K positions. It therefore validates both nibble-pair byte placement
and group32 scale placement without relying on model output hashes.

## Next gate

Wire the accessors into an MFMA64 gate/down oracle and require complete routed
stage agreement against the logical/raw path before changing the production
selector. Decode GEMV/grouped kernels remain untouched until the MFMA oracle
passes.
