# TP8 G1 wo_a tier screen (2026-09-08)

Parent checkpoint: 7de99bebfe. Production selector and live service unchanged.
Original weights, TP8/EP1/no-A2A/native AR. Existing BS1 opt-in remains enabled.
Checked amd-smi process ownership and idle service before isolated GPU4 test.

## Component screen

`scripts/rocm/bench_dsv4_tp8_woa_tiers.py` runs existing grouped HIP specialization
directly, without widening the production shape guard. Five ABBA cycles,
64-node graph bursts x20 replays/sample. Synthetic BF16 input/weights.

| M | einsum median us | grouped median us | reduction |
|---|---:|---:|---:|
|1|30.2451|6.9203|77.1%|
|2|30.3738|8.0965|73.3%|
|4|30.4361|14.1362|53.6%|
|8|30.8316|24.5869|20.3%|

Every shape:100 input mutations, weights changed every25,10 replay checks per
mutation;100/100 finite and replay-exact. Entire-tensor equality with einsum:
70/59/26/5 of100 at M1/2/4/8. Max relative L2 versus FP32 reference:
0.001788/0.001741/0.001723/0.001697. This is not old-GEMM bit-exact arithmetic.
Raw artifact:/tmp/dsv4_tp8_woa_tiers_20260908.json.

## Unchanged-service C4 baseline

Service PID2268456, loopback30011. Existing diverse concurrent harness,
request-count4, seed20908,256 forced output tokens,three waves.
Selected corpus saved:/tmp/dsv4_tp8_c4_workload_20260908.json.
Result:/tmp/dsv4_tp8_c4_control_20260908.json.

- Cold aggregate56.07 tok/s (18.26s); warm181.284/181.092 tok/s (5.65s).
- Common resident-window187.36/187.42/187.43 tok/s; historical JSON key is
  `resident_bs32_tok_s`, but actual concurrency here is4, not32.
- All12 requests completed256 tokens; France first9 exact all3 waves.
- Full hashes are NOT all cross-wave exact, including code requests. France
  is forced beyond its EOS for load here, so its post-EOS hash is not a quality
  criterion. No new production math was loaded in this control.
- This is not a semantic acceptance suite; code answers need separate normal-EOS
  inspection and fixed-prefix comparisons before accepting a candidate.

## Decision

M2/M4 warrant an opt-in native-decode-only service experiment. M8 is secondary.
At M4 the isolated difference is16.30us/layer, about0.70ms over43 layers if fully
critical, versus measured21.21ms scheduler step: an approximate3.3% time budget,
not a promised E2E gain. Do not claim the53.6% micro improvement as service gain.
Next: preserve this exact request manifest, add a separate guarded selector,
run full-service ABBA and correctness including normal EOS and batch tier changes.
No production default changed; no new E2E improvement claimed by this screen.
