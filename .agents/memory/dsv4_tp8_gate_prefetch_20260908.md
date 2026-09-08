# TP8 I256 gate row-prefetch standalone screen

Base5aa2f5a971; original native TP8 weights untouched. Production row-prefetch
wrapper only admits I512/TP4; underlying templated HIP kernel accepts I256.
Independent oracle calls the JIT entry directly, without widening production
guards. Current TP8 A4/R2/W8/G832 and LDS LUT retained. Candidate changes to
existing row-prefetch/DPP gate implementation; no down-kernel change.

amd-smi found no external GPU PIDs. Service restoration completed beforehand:
PID2788470, no realtime trace environment, C32 E2E972.8648/resident1019.8871,
all12prefix transition probes exact. No concurrent HTTP load. Oracle GPU4 only;
synthetic packed weights transient (~256MiB gate; ~128MiB down when full), no
persistent weight cache or KV capacity change. Initial script import failed
before GPU tests; neighboring-module import corrected and rerun successfully.

Five ABBA cycles,100graph replays/sample,20warmups.100input mutations and
ten replay stability checks per mutation. Synthetic unique top6 per token,
three routing concentrations. Baseline is actual non-DPP TP8 grouped kernel.
Input/scales vary; full-stage run mutates per-element E8M0 scales122..127.

Gate-only initial screen (uniform weight scales; actual active experts134/99/32):
155.807->146.980,133.791->127.947,102.687->99.115us.

Post-sort gate->intermediate quant->down partial->fixed reduction chain:

|Active experts / A4 scans|A us|B us|Saved us|
|---|---:|---:|---:|
|133/133|279.5235|275.1390|4.3845|
|104/105|249.1211|241.7887|7.3324|
|32/61|186.4588|183.1654|3.2934|

All runs100/100 exact gate and final BF16 outputs, max_abs0, stable replay.
This is not an external semantic oracle; weights and routing are synthetic.
Sorting and initial hidden-state quant are excluded and unchanged, so do not
call this a measured entire service/routed pipeline improvement. Gate-only and
full runs consume RNG differently; compare A/B within each, not across tables.
No production selector was changed. Next: narrow default-off TP8 native M32
selector, preserve I512 routes and C1; E2E ABBA with real coding manifest and
prefix/France correctness. Do not claim the micro gain as a model speedup.

Artifacts:
/tmp/dsv4_tp8_gate_prefetch_screen_20260908.log
/tmp/dsv4_tp8_gate_prefetch_full_20260908.log
Script: scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py (--full for chain).

## Default-off service experiment started (pending)

SGLANG_DSV4_GFX90A_TP8_M32_GATE_PREFETCH uses the DSV4 forward scope for
native/decode/batch32/gfx90a gating. AR and gate flags are independent: gate-only
scope cannot enable legacy AR. AIter additionally checks TP8, exact hidden,
topk and both weight shapes, A4/R2/G832/LDS, no MFMA prefill. Public row-prefetch
wrapper adds only M32/I256/G832 to existing accepted shapes; DPP-only guard and
TP4 down paths unchanged. Existing same HIP template, no new buffers.

Eight CPU tests pass, including actual selector Cartesian cases, scope cleanup
and gate-only/AR independence. Service ABBA now running in exec76271:
prefix /tmp/dsv4_tp8_gate_prefetch_abba_20260908
arms GPA1:0 GPB1:1 GPB2:1 GPA2:0, --france-c32 on every arm.
Each: C1 probes, six32-distinct-code waves,12transition probes,32France sentinels
(correctness only). C4 overlap and legacy AR retained in all arms. Pool131072,
mem0.80, original weights. amd-smi showed no external GPU PIDs. Final control
left active; do not promote until E2E and all correctness gates pass.

## Service ABBA complete

|Arm|E2E tok/s|Resident decode tok/s|
|---|---:|---:|
|GPA1 off|970.64279|1016.76614|
|GPB1 on|980.91967|1027.22807|
|GPB2 on|979.95116|1025.53705|
|GPA2 off|974.05525|1019.79930|

Mean of service medians: E2E972.34902->980.43541 (+~0.83%), resident
1018.28272->1026.38256 (+~0.80%). Both B services exceed both A services.
C1 controls~83.61–84.17, candidates~84.08–84.24tok/s; no observed C1 regression,
but do not claim a C1 speedup from a selector which excludes C1.

All four arms: France C32 32/32 through EOS exact (128/128 total), twelve
post-C32 fixed-prefix checks exact (48/48 total IDs/input-logprobs/top20-logprobs),
C1 measured full completion hashes and initial fixed-prefix probes match
reference. Full diverse C32 completion bitwise parity remains unproven and is
not implied by these bounded checks. Candidate logs344hits=43layers x8ranks;
graph capture and request transitions succeeded. Graph0.62GB/GCD, available
28.57–28.63GB, pool131072 unchanged. No persistent weight cache added.

Retain default-off TP8 M32 option as a small validated E2E gain; no TP4,
prefill, speculative or C1 selector expansion. Final service is control/off,
PID2821603. Legacy AR and C4 overlap remain enabled as in every arm. Full
ABBA results and micro samples are in adjacent JSONs. Broader goal stays active.
