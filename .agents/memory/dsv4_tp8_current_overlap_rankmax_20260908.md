# Current TP8 decode rank-max profile after accepted overlap experiments

Base `b451b9f5da`; eight GCDs, native AR TP8/EP1/no-A2A, original checkpoint.
Pool1048576 and mem0.96 retained. Shared-gate fusion off. Existing M32 attention
overlap, legacy AR and gate-prefetch remain on; row-stable prefill remains on.
AMD-SMI showed no external GPU owners before replacing the baseline service.

Only diagnostic environment changed: layer20 realtime trace, graph-only,
LOG_EVERY16. Existing nonblocking pinned readback, no per-step host wait added.
One256-byte timestamp buffer per rank, no persistent weight cache.

C1: 144 complete eight-rank samples, zero incomplete/invalid.
C32: one distinct-code warm wave discarded before defining log bounds; then
two32-request256-token waves. 32 complete eight-rank samples, zero incomplete
or invalid. Replay log does not encode graph tier; these are samples from the
C32 workload window, not a proof that every sampled replay remained tier32.

| Rank-max median span, us | C1 | C32 workload |
|---|---:|---:|
|Layer20|293.36|737.20|
|Attention MHC/norm|27.52|50.16|
|Attention preparation|81.52|150.32|
|Attention core|29.44|49.92|
|Output projections + collective|39.84|113.60|
|FFN MHC/norm|25.04|51.20|
|MoE + collective|103.52|338.24|
|Router, slots16–17|7.36|28.64|
|TopK, slots17–18|14.88|12.48|
|Routed expert, slots18–19|49.60|252.96|
|Final collective/arrival, slots23–24|24.08|40.48|

Do not sum independently selected rank-max medians. All deltas use timestamps
within each device, tick factor0.04us as in the existing calibrated diagnostic.
Markers perturb timing; instrumented HTTP rates are not throughput checkpoints.
Prior C32 prepare227.68us/layer832.16us profile predates overlap and used a
different pool. The new data supersedes its budget; this is not a controlled
measurement attributing all reduction to any one optimization.

## Correctness

France exact; six measured C1 full256-token sequences match reference. Six
fixed-prefix next IDs, input logprobs and top20 output logprobs match the
untraced control. All96 C32 warm/measured requests finish length256; measured
cross-round exact19/32. One-wave warm exact32/32 is vacuous and not evidence of
cross-round stability. No numerical kernel changes in this experiment.

## Direction audit

Routed stage remains largest, but existing down-consumer quant fusion is not
an untested opportunity: memory lines2820 onward in
`dsv4_gfx90a_experimental_switches.md` records the exact TP8 combination,
~7.7% micro gain, only~1.1% service gain plus873.4tok/s outlier, and withdrawal.
TP4 K512 had a separate reachability bug and later true negative; neither
justifies blindly switching the old TP8 candidate back on. No flag changed.

Next bounded diagnostic should split the current113.6us output-projection /
collective boundary into wo_a, wo_b, and collective/arrival before choosing a
new candidate. Avoid reopening withdrawn preparation/down-prefetch trials.

Artifacts prefix `/tmp/dsv4_tp8_current_overlap_marker_20260908`:
`_phases.json` records exact log ranges; `_c1_rankmax.json` and `_c32_rankmax.json`
contain all raw ticks; `_c1.json`, `_c32_warm.json`, `_c32.json` retain request
evidence. Parser `scripts/rocm/summarize_dsv4_realtime_rankmax.py`.
