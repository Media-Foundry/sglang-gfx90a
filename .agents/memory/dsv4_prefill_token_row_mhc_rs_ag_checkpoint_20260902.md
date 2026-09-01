# DSV4 TP4 large-prefill token-row MHC checkpoint (2026-09-02)

## Question

Can the TP4 output all-reduce be split into reduce-scatter/all-gather so each
rank evaluates MHC post/pre and RMSNorm only for its owned token rows?

This is distinct from the rejected hidden-dimension shard: each owner receives
complete H4096 rows, so MHC statistics require no extra small collectives.

## Oracle

`scripts/rocm/bench_dsv4_tp4_token_row_mhc_oracle.py` now has an optional
`--include-reduce-scatter` arm:

- A: AIter stock all-reduce, then full-row MHC on all four ranks;
- B: AIter first-dimension reduce-scatter, owner-local MHC on M/4 rows, then
  AIter all-gather of the normalized full rows.

X is rank-distinct; residual/post/comb/Fn/norm metadata is replicated exactly.
The first draft accidentally made all metadata rank-distinct and correctly
failed the equality gate.  The accepted measurements below use rank-distinct X
only.  AIter's Python reduce-scatter writes its explicit output argument and
returns `None`; the oracle consumes that output buffer directly.

All GPU tests used physical GCDs 4,5,6,7 after `amd-smi process` checks.

## Results

Two complete MHC boundaries, slowest-rank time:

| rows | A: AR + full MHC (us) | B: RS + local MHC + AG (us) | saving | speedup |
|---:|---:|---:|---:|---:|
| 2048 | 3821.900 | 2110.778 | 1711.123 us / 44.8% | 1.8107x |
| 2304 | 4324.266 | 2204.630 | 2119.636 us / 49.0% | 1.9614x |

M2304 used seven-round A/B/B/A rank-max timing, 1000 HIP Graph replays, and
rank-distinct inputs.  Eager and graph outputs were bitwise exact.  The M2304
local-compute-only two-boundary median was 752.101 us; RS+AG publication added
1452.529 us but still remained far below duplicated full-row MHC.

## Cross-boundary owner-state validation

The first result used two independent boundary states.  A stricter graph now
feeds boundary 0's owner-local residual/post/comb directly into boundary 1,
without gathering those state tensors.  Only the normalized H4096 activation
is republished for the intervening TP compute.  Every graph replay checks both
the gathered activation and all three local state tensors against the matching
slice of the full-row reference.

At M2304, 1000 graph replays and seven-round rank-max ABBA remained bitwise
exact:

```text
A: stock AR + full MHC, two chained boundaries = 4368.522 us
B: RS + owner-local MHC + AG, chained state    = 2229.513 us
saving                                         = 2139.009 us (49.0%)
speedup                                        = 1.9594x
```

This proves residual/post/comb do not need to be gathered between layers.  A
production path must keep token ownership stable and defer both the attention
and MoE output all-reduces; enabling only one boundary would force an expensive
state gather or break the chain.

## Decision

This passes the component continuation gate by a wide margin.  It is not yet a
service result.  Production integration must remain TP4/EP1, prefill-only,
M divisible by four, and preserve stable scheduler row ownership.  It must
verify whether one or both cross-layer MHC boundaries can consume owner-local
state before republishing full rows.  France, real heterogeneous C1/C32, and
decode negative controls are mandatory before enabling a default.
