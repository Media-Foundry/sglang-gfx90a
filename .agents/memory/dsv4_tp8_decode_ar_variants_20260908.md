# TP8 decode AIter AR old/new screen

Base8319c4cb9b; service2727188 unchanged (prep fusion off, M32 overlap on).
Checked amd-smi: no GPU PIDs outside this service tree. Eight-rank isolated
torchrun with HIP_VISIBLE_DEVICES=0..7, numactl --interleave=all, OMP threads1.
No concurrent service requests; no production/library edits or recompiles.

Current service uses AIter custom AR. The available512K/1M block-count hooks
are TP4-only. Do not apply those switches to TP8 and assume they took effect.

New standalone script bench_dsv4_tp8_decode_ar_variants.py compares current
use_new=True (A) with existing use_new=False (B). It uses direct HIP allocation
for registered input, deliberately avoiding caching-allocator IPC offset0 bug.
Communicator max_size1MiB; no model/cache allocations and no persistent changes
to the serving KV pool. Standalone process/context memory is transient.

Five ABBA cycles,200graph replays/sample,20warmups, slowest-rank time per sample.
100independent rank-local input mutations (normal and integers), ten additional
candidate replays per mutation. Every rank: A/B exact100/100, max_abs0, replay
stable. This is relative to existing AR, not a teacher-forced model oracle.

|Shape BF16|Bytes|Current trimmed us|Old trimmed us|Decision|
|---|---:|---:|---:|---|
|[1,4096]|8192|15.10644|15.14905|Neutral/slightly slower; keep current|
|[32,4096]|262144|30.43575|27.88494|Candidate saves2.55081us (8.38% latency)|

No E2E claim. Even86collectives saving2.55us would be only~0.219ms/token-step;
actual applicability and overlap must be measured, not multiplied into a claim.
Next: optional default-off SGLang adapter restricted to TP8/gfx90a native DSV4
decode M32 BF16, preserving C1/prefill/speculative. Benchmark same real32coding
manifest with independent-service ABBA and fixed-prefix/transition correctness.
Do not globally replace AIter or change its default use_new argument.

Raw rank-max samples and all-rank witnesses are in:
/tmp/dsv4_tp8_ar_variants_m32_20260908.log
/tmp/dsv4_tp8_ar_variants_m1_20260908.log

## Service experiment launched (pending; do not promote)

Default-off SGLANG_DSV4_GFX90A_TP8_M32_LEGACY_AR wraps the DSV4 model forward
in a ContextVar scope. The scope requires HIP/gfx90a, native spec_algorithm,
decode mode and batch32. The AIter adapter additionally requires world8,
contiguous BF16[32,4096] and quantization off. It forwards the original output,
registration and quantization arguments unchanged; no additional GPU buffers.
First actual selection logs once per rank. Context is restored in finally.

Three CPU tests passed: eligibility combinations, actual adapter forwarding,
exception restoration and speculative exclusion. Production E2E pending.

Live sequence exec10114:
--arms ARA1:0 ARB1:1 ARB2:1 ARA2:0
--prefix /tmp/dsv4_tp8_m32_legacy_ar_abba_20260908
state file: /tmp/dsv4_tp8_m32_legacy_ar_abba_20260908_state.json
First verified live service ARA1 PID2741665. Subsequent parent IDs are in state;
verify PID/session before polling/restarting. Explicit interleave-all retained.
Uses existing C1 reference and32distinct coding prompts; six C32 waves/arm,
fixed-prefix plus transitions each arm. Leaves final control active. Accepted
M32 attention overlap stays on; withdrawn C1 prep fusion remains off. Pool131072.

Partial service results (not acceptance):
- ARA1: E2E965.09976, resident decode1010.11717tok/s; C1 case medians
  84.1859/84.2148/84.1912. All12transition probes exact.
- ARB1: E2E976.86647, resident decode1022.62180tok/s; C1 case medians
  83.0177/83.1690/83.1476. All12transition probes exact. Eight actual legacy-AR
  selection logs, graph capture succeeded. C32 ~1.2% faster in this pair but C1
  independent-service variation is nontrivial; await complete ABBA.
- ARB2 is now loading, verified live PID2756481; same exec10114 owns sequence.

## Completed service results

|Arm|Client E2E tok/s|Resident decode tok/s|
|---|---:|---:|
|ARA1 off|965.09976|1010.11717|
|ARB1 on|976.86647|1022.62180|
|ARB2 on|972.02285|1017.32884|
|ARA2 off|961.84421|1006.54998|

Mean of independent-service medians: E2E963.47199 ->974.44466 (+1.1389%);
resident1008.33358 ->1019.97532 (+1.1546%). All four arms pass C1 France,
fixed-prefix probes and12transition probes each. Full heterogeneous C32
completion bitwise parity is NOT established by those bounded probes.

Extra independent ARB3: E2E977.22969, resident1022.95830. C1 medians
83.15990/83.15734/82.93574. Its12transition probes pass. C32 repeated France
sentinel: both final control ARA2 and candidate ARB3 answer-through-EOS32/32
exact; each exercised253decode steps at M32. Repeated France is correctness
only, excluded from reported throughput. Graph memory remains0.62GB/GCD;
available memory remains28.57–28.63GB. No extra model weights or KV shrink.

Decision: retain opt-in C32 experiment, NOT globally enabled. C1 controls ranged
~83.35–84.21, candidates~82.94–83.75; despite the strict M32 selector, an E2E C1
non-regression is not proven. Treat this as a C32 profile tradeoff, not a general
C1/C32 default win. Six TP8 unit tests pass. The standing confirmation service
PID2771780 explicitly enables it for further C32 work; repository default False.
Do not mark the broader C1/C32 optimization objective complete.

Confirmation artifacts prefix /tmp/dsv4_tp8_m32_legacy_ar_confirm_20260908_ARB3;
control sentinel /tmp/dsv4_tp8_legacy_ar_france_A_20260908.json.
