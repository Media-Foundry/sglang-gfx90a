# TP8 prefetched gate paired-scale load: rejected micro candidate

Baseline 9a936ab0a6. Native TP8 service PID3223021 unchanged, 1M KV pool.
AMD-SMI process audit found no external owners. Standalone GPU4 only; no
persistent model cache, weight precision change, or service selector change.

The shuffled E8M0 gate/up address formula ends in pack2. For a gate row<I,
its scale offset is even and the corresponding up scale offset is gate+1.
Checked 98,304 pairs (experts 0/1/255, every I256 row and K4096 group).
An oracle-only macro replaces each two byte loads with one uint16 load and
byte extraction, retaining logical-layout fallback and all arithmetic order.

Three synthetic routing/weight/input fixtures, 100 mutation comparisons each:
BF16 intermediate/final and FP32 partial exact, max_abs=0, graph replay stable.
Full chain gate→quant→down→fixed reducer, five ABBA cycles, 100 replay samples,
discard min/max:

| Active/scans | Baseline us | Paired-scale us |
|---|---:|---:|
|133/133|272.7663|277.5804|
|104/105|241.3588|247.2213|
|32/61|182.3361|185.0275|

Latency regresses roughly 1.5–2.4%. Do not integrate into service. Source-level
load count is not a performance argument; no disassembly/resource evidence
was collected to attribute the regression to a particular compiler effect.
Production lacks `SGLANG_FP4_GATE_PAIRED_SCALE_LOAD_ORACLE` and is unchanged.
This does not reject other scale-address optimization or explicit-layout work.

Reproduce using the DS interpreter, repository PYTHONPATH and
`HIP_VISIBLE_DEVICES=4 scripts/rocm/bench_dsv4_tp8_gate_prefetch_screen.py
--paired-scale-load` (invoke the script through Python).
