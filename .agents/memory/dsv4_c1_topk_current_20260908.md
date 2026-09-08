# C1 communication / Top-K path audit

Baseline ae41515b85. No serving code, dependency, weight or KV-pool changes.

The same-day AR old/new experiment already measured C1 at15.106/15.149us,
so it was not repeated. M32-only legacy AR adapter does not affect C1.

The installed AIter exports no `topk_gating`. An initial direct probe failed
with AttributeError; no successful AIter timing is claimed. SGLang catches
this ImportError when initializing `_aiter_topk_gating`, then uses
`moe_fused_gate` for sqrtsoftplus selection. This is an intentional fallback,
not proof of a dispatch regression. The service environment has both legacy
native/packed Top-K switches0 and ROUTER_NUM_WARPS1.

Measured the actual fallback against the existing HIP sqrtsoftplus helper:

| Path | Graph median us |
| --- | ---: |
| Current one-warp Triton fallback | 7.16727 |
| Legacy HIP sqrtsoftplus | 13.78254 |

Seven ABBA cycles,200 replays/sample. GPU4 only, AMD-SMI verified no external
owners before probe. Synthetic BF16 logits with real layer20 checkpoint bias
converted to runtime BF16; scale1.5, renormalize true, scale-on-output false.
100 mutations include ten equal-logit fixtures. IDs match100/100; weights
match2/100 bitwise, max absolute difference4.47035e-8. Candidate replay stable.
This does not prove cutoff membership under arbitrary ties or model quality.

Legacy HIP is about92% slower, so do not enable it merely to remove Triton.
No E2E candidate added. The approximate15us C1 service Top-K marker includes
dispatch/scheduling scope; it is not equivalent to this7.17us isolated kernel.

Script: `scripts/rocm/bench_dsv4_c1_topk_current.py`. Added guards require the
same fallback flags/warp count and fail if future AIter provides the direct
implementation, so a changed environment cannot silently reuse this baseline.
The adjacent JSON contains all measured samples. Existing 1M KV service left
untouched. These data close two redundant C1 tuning avenues; they do not report
a new C1/C32 throughput checkpoint.
