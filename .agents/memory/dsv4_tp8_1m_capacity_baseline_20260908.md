# TP8 capacity-aligned decode baseline (pending)

Previous two goal turns produced isolated negative/marginal kernel evidence,
not new E2E gains. Restore accepted TP8 options and validate the actual large
KV profile before further throughput tuning. Prior~980 C32 E2E observations
used pool131072/mem0.80; do not present them as1M-pool performance.

No production changes. All GPUs idle in amd-smi before launch. Parent2857900
verified running sglang.launch_server; exec session56171. Log:
/tmp/dsv4_tp8_1m_capacity_baseline_20260908.log

Native original checkpoint, TP8/EP1/no-A2A, GPUs0–7, localhost30011,
pool1048576/mem0.96, prefill36864, graph1/2/4/8/16/24/32, NUMA interleave-all.
Accepted BS1 wo_a, runtime-M, C4 M32 overlap, legacy AR and gate-prefetch ON.
Withdrawn attention-prep fusion OFF; down-prefetch production wiring absent.
Both throughput/multi-request profiles enabled, trivial C4 logits skip enabled.
Launch environment confirms C4 graph bound262144 (131072pool used32768).

Pending: load/capture capacity, actual free VRAM/graph memory, C1 France and
fixed-prefix oracle, six C32 diverse coding waves, transition probes. No claim
yet that1M pool can capture with current workspace or sustains prior speed.
Keep same live process through readiness waits; do not restart on timeout.
Pool capacity alone does not establish1M-context correctness or throughput.

## Capture passed; request validation in progress

All8ranks captured tiers1/2/4/8/16/24/32 in12.21–12.34s. Graph memory0.65GB
perGCD, remaining15.58–15.64GB. Runtime confirms max_total_num_tokens1048576
and context_len1048576. No OOM/capacity downgrade observed at capture.
This does not yet test large-prefill workspace at its maximum live shape.

Client controller exec16212 waits on verified server2857900, then serially
runs C1(two measured rounds/task), six diverse C32 waves,12transitions and
C32France. It stops on any failed oracle; does not kill/restart server.
Artifacts /tmp/dsv4_tp8_1m_20260908_{c1,c32,transitions,france_c32}.{json,log}.
C1 France already passed; final speed/parity checks still pending at this note.

## Completed request validation

Client session16212 completed successfully; server2857900 remains resident.
C1 medians82.5033/82.7058/82.7683tok/s; six measured completions match the
131072-pool reference IDs exactly. All six supplied-prefix input logprobs,
output IDs and top20 output logprobs exactly match that reference.
C32 six-wave E2E median980.3196, resident-window1027.3754tok/s;32distinct coding
requests/wave, all256tokens, no speculative counts.12transition probes exact
and32/32France exact. See portable JSON for rounds and artifact digests.

Complete diverse-request cross-round hashes:0/32exact. First divergence ranges
from generated position0 to142. This was already an acknowledged limitation of
the earlier small-pool baseline; these bounded checks do not prove deterministic
full C32 outputs or long-context correctness. Do not describe it as newly fixed.

Compared with the prior gate-prefetch small-pool mean-of-arm-medians980.4354,
the current single-service980.3196 is similar, not a controlled pool A/B.
C1 is lower than earlier~84; cannot attribute causally to pool size without
same-session/independent-service ABBA. No new speedup or code change claimed.
This establishes a capacity-aligned starting point without shrinking KV to
obtain throughput. Future experiments should retain this1M pool unless a
separate explicitly labeled diagnostic needs smaller capacity.
