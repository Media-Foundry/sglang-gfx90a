# TP8 G1 wo_a: real-weight multi-matrix screen

Parent62171a2152. Serving code/flags unchanged; diagnostic service3097771
remained idle during GPU4-only component tests. AMD-SMI ownership checked
before each GPU process. Original checkpoint weights from layers0/6/12/18/20/
24/36/42, rank0 TP8 shards: eight BF16[1,1024,4096] matrices,64MiB total.
This exceeds a single-matrix cache-resident working set. No serving weight cache
added or KV pool reduced. Library workspace/peak device allocation was not
recorded; this is not service-memory acceptance.

Input is32 rows from an existing real layer20 *prefill* activation dump,
reshaped[32,1,4096], not a new C32 decode dump. Perturbations add BF16 Gaussian
noise to activations. Weights stay real checkpoint-derived BF16 in these tests.
No claim of exactness across all weights/inputs or long-context output quality.

## Ordinary API spelling

Seven ABBA cycles,8 independent matrix nodes per graph,50 replays/sample:

| Candidate | baseline einsum us | candidate us | exact mutations |
|---|---:|---:|---:|
|F.linear|32.7203|32.7159|100/100|
|torch.mm|32.7215|32.7063|100/100|
|AIter tgemm|32.7351|32.7347|100/100|

All1000 replay-stable. AIter logs show Torch solution0 fallback, not a CK
kernel. No meaningful gain from spelling changes; do not connect them.

## Actual hipBLASLt instances

The initial2000-instance guard declined2226 supported solutions before timing;
raised the explicit ceiling to4096 and reran. Screened all2226 on the same eight
real weights; ranked by median of three graph samples. Fastest three then
received100 activation mutations,1000 replay checks and seven ABBA cycles.

| Solution | baseline us | candidate us | max abs | max relative L2 |
|---|---:|---:|---:|---:|
|4429|32.7611|18.2185|0.015625|0.000127068|
|5487|32.7797|18.3251|0.015625|0.000119554|
|3936|32.7561|18.3883|0.015625|0.000142004|

All three are0/100 bitwise exact relative to einsum, finite and1000-replay
stable. Fastest initially exact solution3719 took32.8240us: no known exact
speed win from this screen. Full ranked table and ABBA samples are retained.

## Disposition

4429 saves14.54us (~44.4%) in the component. This is a changed floating-point
reduction path, not a change to checkpoint or weight precision. User permits
small drift subject to actual output checks; therefore consider a *default-off*
native TP8/C32/G1 wo_a selector and full real-code E2E test. C1, prefill,
speculative, other TP/tier paths must remain unchanged. Verify library version
and actual dispatch; hipBLASLt solution numbers are not portable contracts.

No E2E gain claimed. Even43 exposed savings would be only~0.625ms per C32
step, and overlap/rank-tail can hide them. Require France, fixed-prefix logits,
real coding outputs and ABBA, plus unchanged1M KV pool and measured workspace.
Do not globally alter AIter tuned CSV or replace generic linear dispatch.

Reproduce with `HIP_VISIBLE_DEVICES=4`, DS Python,
`scripts/rocm/bench_dsv4_tp8_woa_gemm_screen.py --service-pid <idle-service> --output <json>`;
add `--hipblaslt` for the full solution screen. Adjacent JSON stores both runs.
