# Corrected first-MHC service cost measured; exact vocabulary-table oracle wins

Continuation of26d981fe9d. Original V4,TP8/EP1,no-A2A,1M logical KV,32K chunk,
C16 real heterogeneous8K code requests,total131069 tokens,zero prefix hits.

## Uninstrumented service ABBA

Experiment dsv4_first_mhc_service_20260917,run.py session12148 exit0. Three fresh
services A1/B/A2; B has two consecutive three-wave legs, all timed legs3 waves.
Warmup excluded. Route producerON in BOTH arms. A uses previously accepted MHC
policy. B enables COMMON_SMALL and COMMON_UNFUSED, keeping FP32 Fn/20 iterations
at the affected boundaries. No diagnostic probes/markers or global BLAS changes.

- A1:10376.295908 input tok/s
- B1:10320.584363
- B2:10306.722991
- A2:10346.720395
- Control center10361.508152,candidate10313.653677: -0.461849%.
- Control A1->A2 drift -0.285030%; within-leg ranges .061%.. .340%.

Analyzer recomputes every rate from earliest request start to last first token,
validates16 identical manifest inputs,131069 tokens,no cache hits,one output
token,3 waves/leg,1M pool and identical source hashes. All8ranks hit route
producer,premix owner,K32. Timed workloads have fourM32768 forwards/wave. Three
services stopped cleanly. No numerical-equivalence or universal invariance claim.

All France responses contain Paris. One separate128-token code-output wave/arm:
A1/A2 match16/16,B versusA1 only1/16. All16 B excerpts manually read: coherent,
on-topic,no obvious loops/garbling within128 tokens. Their asserted source bugs
have NOT been independently verified; not a full quality-equivalence claim.
Default launcher unchanged. B/start-ar-matrix.sh is an independently measured
corrected reference, not a throughput win. evidence.tar.gz (~2MB) freezes141
files including raw timings,quality,source snapshots and stop proofs.

## New exact candidate: first-layer FP32 mix table

Source DeepseekV4Model.forward: token embedding is expanded to four identical
HC residual streams before layer0. For plain input_ids (NOT input_embeds or
adapters),the unfused first-layer raw mixes depend on token ID and static
embedding/Fn/epsilon only,not position or prefix. Cache only these24 FP32
numbers/token,not normalized activations or any later-layer state.

dsv4_first_mix_table_20260917/oracle.py session38636 exit0 uses actual checkpoint
embed.weight[129280,4096] and layer0 hc_attn_fn. Hashes match the live model;
embedding lookup repeated4 equals all64 captured service suffix residuals.
Build chunks1024 bound temporary repeated activation to32MiB. Table size
12,410,880 bytes (~11.84MiB)/GCD. Build plus whole-vocabulary validation .1503s
in isolated process; not a service-startup measurement.

Every vocabulary row validated between existing three-column and two-column
native FP32 kernels. Table lookup matches captured corrected service mixes.
Seven shapes M1/64/65/1024/8192/32576/32768,reverse rows,and100 HIP graph replays
with changing real code token IDs per shape all pass byte-exact (700 mutations).
Component graph ABBA (no service claims):
M8192: ready-residual native premix3.6161ms -> lookup .01551ms.
M32768: ready-residual native premix14.3681ms -> lookup .04566ms.
Control timing EXCLUDES embedding gather/repeat,so this is not a fake win from
charging the control extra setup. The lookup only replaces first raw premix;
Sinkhorn,weighted mixing,norm and later residual computation remain unchanged.

Not integrated. Next must build from TP-local vocabulary shards,exchange only
small FP32 mix tables,verify padding/global-ID mapping,own cache on model/runner,
invalidate on weight reload/versions,exclude input_embeds/LoRA/spec/AR/V4.1,
and preserve1M pool. Do NOT copy full embedding weights per GCD as the isolated
single-GPU fixture did. Live eight-rank byte-exact oracle before service ABBA.
Expected E2E budget is only the first-boundary cost (~tens of ms/wave),not300x
whole-model speedup. Could recover the~0.46% correction cost; unproven until E2E.
