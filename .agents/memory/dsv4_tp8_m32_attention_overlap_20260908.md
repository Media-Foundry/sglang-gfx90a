# TP8 native C4 M32 attention overlap — active screen

Goal: optimize native TP8 C1/C32 without consuming KV pool for weight caches.
Current profile summary is `dsv4_tp8_current_decode_rankmax_20260908.md`.
After trace, narrow default-off `TP8_M32_ATTN_MULTISTREAM` was added to existing
HIP prepare implementation. Constructor admits TP8 C4 streams; execution
requires unified-KV, M32, batch32, native decode and graph capture. Generic
overlap flag cannot bypass the narrow TP8 gate. C1 and eager prefill stay serial.
No checkpoint math/kernel changed; parallel scheduling can change ordering.
112 CPU predicate combinations passed. Existing user replay hooks in model
are unrelated, were preserved, and MUST NOT be staged with this experiment.

Candidate B parent2654269 is stopped. Six fixed32 real coding request waves:
complete-request median891.027 tok/s, trimmed892.307; resident decode median
937.354 tok/s. Prior runtime-M control was922.033/963.292, but it was earlier,
not a formal ABBA. All192 requests completed256 tokens / length. As in the
baseline, 0/32 completion hashes identical over all waves; no new claim of
full C32 bitwise or semantic validation. Candidate remains unaccepted.

C1 France exact; three measured completion hashes match baseline and six
fixed-prefix probes match IDs/input logprobs/output top20 exactly. Rates
79.722/64.531/71.244 tok/s were lower/variable although C1 isn't selected.
After C32, twelve fresh-cache fixed-prefix probes also exactly matched.
Cannot attribute C1 timing differences to candidate without closing control.

VRAM: graph reports0.62GB, available28.57–28.63GB, max_total_num_tokens131072
unchanged. amd-smi reported no external GPU processes. No weight cache added.

Closing uninstrumented control A2 with flag0 is now starting/running:
parent2661758, `/tmp/dsv4_tp8_m32_attn_overlap_A2_20260908.log`.
Client sequence waits for readiness then C1 correctness and six C32 code waves.
Artifacts when complete:
`/tmp/dsv4_tp8_m32_overlap_c1_A2_20260908.json`,
`/tmp/dsv4_tp8_m32_overlap_c32_A2_20260908.json`.
Candidate artifacts use B instead of A2; transition B artifact also exists.
Do not restart an alive control because a tool observation times out.

Next: inspect A2 results, reject/remove temporary gate if no advantage;
if promising, require true same-candidate ABBA and stronger C32 decode parity.
Keep runtime-M repair and user's unrelated worktree changes.

## NUMA confound discovered during closing control

The diagnostic/B/A2 restarts copied Python argv/env but omitted the original
launcher's `numactl --interleave=all` prefix. `/proc/2662046/numa_maps` showed
3513 default mappings,304 prefer=static:0,4 prefer=static:1,3 local; no interleave.
The standard TP8 launcher defaults to interleave. Thus B/A2 are not the same
host-memory policy as the earlier runtime-M baseline. Do not claim a production
regression or gain against that earlier baseline from these runs. The marker
results are also measurements under default NUMA placement, not interleave.
Both candidate and closing control showed timing variability; its causality
is not proven merely by discovering this config mismatch.

Restore explicit numactl --interleave=all before further performance comparison.
A2 parent2661758 stopped normally after all requests completed. A3 interleave
control log: `/tmp/dsv4_tp8_m32_attn_overlap_A3_interleave_20260908.log`.

## Unified-policy ABBA now in progress

A3 interleave control completed: C1 case medians83.681/83.662/83.717 tok/s;
C32 complete-request median919.607, resident decode959.959 tok/s. Eight rank
numa_maps each contain interleave:0-1. This reproduces the earlier speed much
more closely, but is not proof that NUMA was the only cause of prior variance.

`scripts/rocm/bench_dsv4_tp8_m32_overlap_arms.py` now runs independent B3/B4/A4
services after A3, explicitly restoring numactl policy each time, same env/argv,
same32 coding inputs/six waves and C1 two measured rounds/task. Each arm also
checks12 fixed-prefix transitions after C32. Failing readiness checks leave the
live process available for inspection, never automatically restart it.
Prefix `/tmp/dsv4_tp8_overlap_interleave_20260908`; `_state.json` records live PID,
arm, policy and completed artifacts. B3 initially parent2677031. Root exec
session89695 owns the sequence. Do not run simultaneous GPU tests or another
benchmark against these arms. Last A4 service will remain resident.

## Interleave ABBA completed; C32 semantic check pending candidate

Same-policy independent services:

| arm | complete-request median tok/s | resident decode median tok/s |
|---|---:|---:|
|A3 off|919.607|959.959|
|B3 on|963.401|1008.315|
|B4 on|957.789|1002.260|
|A4 off|918.728|959.229|

Mean of the two candidate service medians versus the two control service
medians: decode +4.762%. C1 candidate medians remained about84 tok/s.
Each B3/B4/A4 arm passed France C1, six fixed-prefix logprob probes and twelve
post-C32 probes. The source predicate unit test passes. KV pool131072 intact;
graphs about0.62GB/GCD. No default was enabled.

Additional `check_dsv4_france_c32.py` correctness-only repeated sentinel on A4:
32/32 first-nine tokens exact through EOS. Forced256 length is used only to
exercise co-resident graph; post-EOS text is not judged. Decode moments show
253 steps/8096 token rows (M32 each), common resident window7.807s. This is NOT
a diverse throughput benchmark; the actual ABBA performance corpus is32
distinct coding prompts. Raw `/tmp/dsv4_tp8_france32_A4_20260908.json`.

A4 parent2691502 has now stopped normally. Extra candidate B5 is running via
the same helper (root exec session37845), same prefix, --arms B5:1. It first
repeats C1/C32 and transition checks; then run the C32 France checker against B5
and compare the32 first-nine answers with A4 before accepting the service.
The reused prefix means `_state.json` now tracks B5 only; historical per-arm
C1/C32/server/transition artifacts are still intact. Use a unique state prefix
for future sequences to avoid replacing the sequence summary.

## Accepted opt-in checkpoint

B5 parent2699916 remains live, interleave all, marker off, candidate flag1.
Third independent candidate: C32 complete-request957.026 / resident1002.103
tok/s; C1 task medians83.975/83.940/84.032. Six supplied-prefix probes and12
post-C32 probes all exact. C32 France checker passes32/32 first-nine tokens
through EOS, same answers as A4. Both runs report253 decode steps and8096 rows,
confirming M32-resident decode. Full post-EOS output is not a quality oracle.
This does NOT establish bitwise parity of all long, diverse C32 trajectories;
the existing across-wave hash variation is still present and disclosed.

Keep the new switch default-off globally; the current winning native TP8
service explicitly enables it. Native short-context C32 ABBA gain is4.762%;
no general long-context/C32 or1M-pool performance claim. No precision/weight
change or weight caching. B5 graph0.62GB/GCD and131072-token pool unchanged.
The existing short C1 route stays unaffected. Do not extrapolate C1 multistream
rejection to C32; the M32 screen demonstrates a distinct positive regime.

Portable per-arm timing/correctness summary and32-answer witnesses are in the
adjacent JSON. Scripts preserve all detailed raw artifacts under `/tmp`.
The model file still includes unrelated user's CK replay hooks: stage only the
two attention-overlap hunks, not the replay hooks.
