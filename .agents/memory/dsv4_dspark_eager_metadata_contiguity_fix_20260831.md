# DSpark eager draft metadata contiguity fix

Date: 2026-08-31

## Bug

For gamma shorter than the checkpoint block, DSpark selects the draft columns
from the full `[bs, gamma+1]` verify window.  At gamma one,
`positions_2d[:, :1].reshape(-1)` retained stride 2.  CUDA-graph staging copies
this view into contiguous graph buffers, hiding the defect; an eager DSpark
forward passed it directly to the AOT fused norm/RoPE kernel, which correctly
rejected it:

```text
Tensor<32>[strides=<2>, dtype=int64] ... is not contiguous as expected
```

Both draft positions and cache locations now go through
`flatten_draft_window`, which materializes the selected columns before
flattening.  This code is under the DSpark proposer only; native AR is not
touched.

## Verification

- CPU regression test verifies a `[4,3][:,:1]` selection becomes stride-one
  contiguous with values `[0,3,6,9]`.
- Target test module: `2 passed`.
- Fresh eager TP4/EP1/no-A2A DSpark service on physical GCDs 4--7 completed 32
  real coding requests and produced the M64 C128 production replay payload.
- Eager France run: first-nine exact, semantic Paris true, all 32 requests
  completed 32 tokens with `finish=length`.

Artifacts:

```text
/tmp/dsv4_pair_dump_server.log
/tmp/dsv4_pair_dump_trigger2.json
/tmp/dsv4_dspark_eager_contig_fix_france.json
```
