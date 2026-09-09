# TP8 DSpark C32 >=2000 tok/s — active

Requested objective: at least2000 output tok/s at C32 with TP8. Preserve
original checkpoint weights, full-target verification and native AR isolation.
Keep the1M logical-token pool unless a capacity tradeoff is explicitly agreed.
Use real diverse code requests, held fixed across arms; optimization screening
uses one ABBA per candidate per the user's latest instruction. No approximate
anchor-only target path and no fabricated/simulated acceptance results.

Starting evidence: prior TP8 communication+C128 full-target profile ~932 tok/s;
optional refined-C4 screen937.900427 tok/s versus matched control922.802887,
observed+1.636052%. These are common resident decode intervals excluding
prefill/drain, not whole-wave wall time. Historical TP4~1.5k used a rejected
target approximation and is not an equivalent control. New2k target remains
unachieved; no promise that a settings-only change can double throughput.

## First diagnostic, not a performance screen

Revalidated HEAD5e35016f64 and live service165825; amd-smi showed only owned
GPU PIDs. Existing DsparkInfoDumper supports core, step CPU/GPU, draft GPU,
target-verify GPU and request records. Reuse it instead of editing the hot path.
It records only attention TP rank0, so its event durations are NOT independent
eight-rank maxima or a system-wide kernel-duration sum. Diagnostics add event
and copy overhead; their tok/s must not be compared as an optimization win.

Controller `/tmp/dsv4_tp8_dspark_2k_phase_profile.py`, exec session71525:
- Stops only the owned service after another amd-smi ownership check.
- Restarts the conservative full-target profile with observer components on.
- Fixed real-code manifest `/tmp/dsv4_open_code_pd_20260908/decode.json`.
- One C32x256 warm wave, then C32x1024 diagnostic wave, natural EOS.
- Saves only extracted dspark_info_record payloads from server_info, not
  arbitrary server environment/configuration secrets.
- Preserves warm records separately, allowing subtraction by forward_ct.
- Finally restores the profile with observers off, C4/refinement off.

Artifacts `/tmp/dsv4_tp8_dspark_2k_phase_profile_20260909`.
No optimization result yet. Next: close draft/target/host-step budgets for
actual C32/M128 rows, then choose a structural candidate. Exact ragged verify
or draft-budget changes require checking target-verification semantics and
realized acceptance, not merely raising a block-size setting.

## Phase diagnostic completed

Controller produced warm and profile records, then began restoring service
210544 with observers OFF. Profile request output passed the repetition gate;
its911.18 tok/s is diagnostic, not a performance comparison. Subtracting warm
records by forward_ct111 leaves137 C32 records, all static M128; excluding
eight edge records each side leaves121. Median /10%-trimmed mean in ms:

| Segment | Median | Trimmed mean |
| --- | ---: | ---: |
| CPU step interval |95.8250|95.7400|
| GPU step |95.4449|95.3661|
| Draft |10.4842|10.4876|
| Target verify |82.3529|82.3191|

Median GPU residual2.5495ms. Mean observed sum of request acc_len87.9091 per
step; event-derived diagnostic throughput922.784 tok/s. At unchanged accepted
output,2000 tok/s requires43.9545ms total step, not merely a few microseconds
off a tail kernel. Target accounts for roughly86% of the observed GPU step.
These rank0 event boundaries include waits and are not kernel-duration sums;
request-detail recording adds D2H/event overhead. Do not attribute every
CPU/GPU difference to uninstrumented scheduler overhead.

Historical review: TP4 forced-budget pruning retained the same graph tier and
lost acceptance without reducing step time. A future exact compact candidate
must prove both actual row-count and graph-tier reduction. TP4's old~1573
segment profile belongs to the prior anchor-dependent profile, not the new
full-target TP8 baseline. Do not reuse its kernel budget as current evidence.

## Next GPU diagnostic queued

`/tmp/dsv4_tp8_dspark_2k_kernel_trace.py`, exec session63235, waits for the exact
phase controller202141 to finish restoration before using service210544. It
rechecks amd-smi ownership, verifies observer env OFF, warms the same real C32
requests, then records eight CPU/GPU profile steps by stage. It does not
restart the service or modify model code. Artifacts:
`/tmp/dsv4_tp8_dspark_2k_kernel_trace_20260909`.
Diagnostic times are not E2E performance claims. Next action is to identify
the target critical kernels and validate a specific structural change with
correctness + single ABBA. No speed optimization has yet been accepted for
the new2k objective.

## Kernel trace failed; use existing async markers

Legacy CPU+GPU /start_profile produced eight EXTEND traces but no DECODE
trace. queue_interposition.cpp signal-handler wait counters kept increasing
without value changes; a separate bounded health request timed out. Do NOT
read those waits as model-kernel costs. This repeats the instrumentation
failure already recorded in dsv4_tp8_bs1_marker_profile_20260907.md; that
history should have been checked before attempting this profiler route.

Only owned trace controller214097, its benchmark child, and service210544's
tree were terminated after amd-smi ownership validation. Controller63235
exited143 (cancelled, not complete). Recovery script
`/tmp/dsv4_tp8_recover_profiler_20260909.py`, session17212, stopped all tracked
processes and started service219876 without the profiler. Recovery artifacts
`/tmp/dsv4_tp8_profiler_recovery_20260909`. No hardware fault established.

Replacement uses the already-implemented nonblocking RealtimeTraceReadback
and s_memrealtime markers (not the older blocking readback). Native TP8's
successful20260908 marker history was inspected; the markers add overhead
and their cross-stream spans include waits. This is not a throughput test.

Queued controller `/tmp/dsv4_tp8_dspark_2k_marker20.py`, session82254, waits
for recovery controller219479 to finish. It then verifies ownership, enables
only layer20 markers, graph-only, every8 replays; C4 remains OFF. One real
C32x256 warm wave, one diagnostic wave, save log byte offset at their boundary.
Finally restores the profile with all marker selectors OFF. Artifacts:
`/tmp/dsv4_tp8_dspark_2k_marker20_20260909`.
Require complete monotonic coarse samples matched by replay ID across eight
ranks; never subtract raw clock values from different GCDs. No accepted new
optimization or2k result yet.

## Layer20 markers completed; first grid candidate screened

Marker controller82254 completed both waves and restored service234817.
No invalid marker warnings. Excluding the first pending warm readback leaves
12 replay-ID-matched eight-rank samples. Rank-max median spans (us): layer
1956.48; attention MHC117.84, prepare210.24, core236.56, output/collective
154.64; FFN MHC118.56; MoE/collective1138.88. Dual-stream MoE slots18->19
(actual routed experts call)1011.28; router42.16, TopK15.92, final collective
71.92. These independently selected medians must not be summed as a critical
path or extrapolated to all43 layers without further evidence. Portable data:
`dsv4_tp8_dspark_2k_marker20_20260909.json`.

Concrete selector gap: aiter.py's use_m128_decode_geometry only accepts TP4
w13[256,1024,2048]/w2[256,4096,256]. TP8's I256 shape fails it and executes
the M>=128 defaults G416/D312, ignoring configured decode G832/D832. This is
not proof that the larger grid is faster; it motivates an independent oracle.

Added standalone `scripts/rocm/bench_dsv4_tp8_dspark_m128_geometry.py` for I256,
single ABBA, same A4/R2/W8/LDS lookup and fixed-slot FP32 reduction. Synthetic
balanced/skewed unique-top6 routes, random nonconstant scales; NOT captured
real routing. Initial input quant and sorter are excluded from timed stage.
Both distributions pass100 activation/router-weight mutations (metadata fixed
within each distribution) and1000 graph replays, all intermediate/output
tensors exact between grid choices.

GPU4, with owned service stopped, single ABBA results (us):
- Balanced: A824.058380 /B802.546234 /B802.731857 /A824.357605.
  A824.207993, B802.639046; speedup2.687%, time saving21.568947us.
- Skewed: A792.184753 /B757.102966 /B756.333313 /A792.146301.
  A792.165527, B756.718140; speedup4.684%, time saving35.447388us.

Artifacts `/tmp/dsv4_tp8_dspark_2k_geometry_20260909`, controller
`/tmp/dsv4_tp8_dspark_2k_geometry.py`, session7744. Both oracles completed;
restored_control service243031 is starting. No production selector changed
yet. Next: default-off TP8 DSpark-target/M128/I256-only selector, guard tests,
then real-code C32 ABBA. This small candidate alone cannot reach2k.
