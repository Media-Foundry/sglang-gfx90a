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
