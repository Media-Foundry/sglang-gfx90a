# Router selected-weight gather: exact small component win, no E2E claim

Parent2bf0d2104d, baseline service3391501 unchanged with1M KV. GPU4 only;
AMD-SMI audit showed no foreign GPU PIDs. No production selector/default/kernel
modified. No weight cache or model allocation added.

## Audit before experiment

Current generic Triton sqrtsoftplus router does six max/min selections and six
full-row one-hot sums to retrieve the selected activation weights. Historical
TopK+sort fusion failed, but its notes describe a prior AOT router; that is not
the currently inspected `_router_triton_kernel`. Do not assume the old kernel
or its reduction contract remains the baseline after upstream changes.

The TP8 model caller supplies no non-padded count to TopK; helper guards return
for None. HashTopK's two zero calls therefore do not establish two GPU launches
on this path. Do not remove general padding masks based on their textual count.

## Candidate and validation

An isolated Triton candidate keeps score math, six max/lowest-ID selections,
normalization tree/zero-denominator guard and optional scaling. After selecting
all IDs, one `tl.gather(activated, selected, axis=1)` replaces six one-hot sums.
Output slots6/7 remain zero before summation, preserving the padded8-value tree.

Tested M1/M32, BF16 logits and bias, optional scale1.5 on/off.100 mutations per
shape/mode include exact ties and underflow-to-zero rows. For each, all IDs and
FP32 weight raw-bit patterns match the actual current production Triton kernel
100/100; maxabs0.1000 graph replays exact and allocated memory stable. This is
synthetic router correctness, not full-model or arbitrary scoring-mode coverage.

Timing:100 operations per captured graph, fiveABBA cycles, ten samples/arm,
trim minimum/maximum. Times in microseconds:

| M / scale | Baseline | Gather | Argmax+gather | Packed-key+gather |
|---|---:|---:|---:|---:|
|1/off|3.14959|2.96199|4.57419|3.13780|
|1/on|3.21758|2.97299|4.57799|3.15599|
|32/off|3.26739|3.10279|4.68059|3.27619|
|32/on|3.33019|3.10739|4.68599|3.27539|

Baseline column is gather's paired run; each later variant's own paired baseline
is preserved in rawJSON. Both alternative selection schemes pass the same exact
checks, but neither beats plain gather. Argmax's compact source does not mean
cheaper generated GPU work. Packed key canonicalizes signed zero and preserves
lowest-ID membership; it is not the old topk8-sort algorithm.

## Budget and decision

Gather saves0.165–0.245us per learned-router layer. Only40of43 layers use learned
TopK (first3 HashTopK are separate), so the optimistic critical-path budget is
6.6–9.8us/token: below0.1% of C1 and about0.02–0.03% of a C32 step. Component
improvement is real in this screen but not evidence for an E2E checkpoint.
Do not immediately spend another multi-process service ABBA or change defaults
for this alone. Preserve the exact gather as an ingredient for a broader measured
router-boundary change; close argmax/packed-key alternatives. No extra standalone
or service job remains after these three completed screens.

Reproduction: `scripts/rocm/bench_dsv4_router_weight_gather.py`; no flag for pure
gather, `--argmax` or `--packed` for rejected selections. Raw output files and all
paired samples are preserved in the companion JSON. No new C1/C32 E2E rate.
