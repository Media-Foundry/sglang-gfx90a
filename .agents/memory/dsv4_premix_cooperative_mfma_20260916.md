# Cooperative FP32 MFMA pre-mix: component win, pending service validation

2026-09-16, base `1b9ec2a063`. No production code or launcher changed.
Accepted E2E checkpoint remains **8964.910269 input tok/s**. The new timings
below are isolated components, not an updated service result.

## Why this follows, rather than repeats, the rejected screen

The scalar-loaded FP32 MFMA was 2.78x slower at M32767. This implementation
changes operand supply: 256 threads cooperatively load contiguous 16-byte vectors
into a bounded LDS tile, then four waves each compute 16 rows and all 24 columns.
Activations stay BF16 in LDS and convert in registers; Fn stays original FP32.
There is no model-weight quantization or full FP32 activation workspace.

Plain row-major LDS was tested first. K32 reduced M32767 pre-mix from 4.965 to
4.528 ms, but M8192 regressed. Padding A by 8 BF16 elements and B by 4 FP32
elements improved the measured result substantially. K64 padded reached 3.096 ms
at M32767. No hardware bank-conflict counters were taken: the padding experiment
is evidence for that layout improvement, not a complete stall attribution.

Finally K-split4/16 with fixed-order FP32 partial reduction trades off performance
and numerical error. No atomics, persistent queue, cross-CTA spin or grid barrier.

## Full pre-mix ABBA

Three graph ABBA cycles, five calls per timing sample. Includes raw projection,
all split reduction and RMS normalization. Sampled real layer0 activations/Fn
expanded/scaled to occupancy, not independent live full-M tensors.

| M | Current paired pre-mix | Split4 candidate | Split16 candidate |
|---:|---:|---:|---:|
| 8192 | ~1.35 ms | 0.538 ms | 0.825 ms |
| 32767 | ~4.964 ms | 3.070 ms | 3.545 ms |

Split4 M32767 latency falls 38.15%; split16 falls 28.60%. These percentages are
not end-to-end gains. Small-M observations are outside the target selector and
must not be presented as a decode improvement.

## Complete post + pre-mix boundary ABBA

Both arms include the accepted exact HIP post. Candidate merges partial reduction
and RMS finish in one kernel, with identical function AST to the validation script.

| M | Current boundary | Split4 boundary | Split16 boundary |
|---:|---:|---:|---:|
|8192|~1.965 ms|1.142 ms|1.429 ms|
|32767|~7.367 ms|5.462 ms|5.952 ms|
|32768|~7.344 ms|5.460 ms|5.960 ms|

At M32767: split4 saves 25.89% of the full boundary, split16 19.17%. Scratch is
12 MiB / 48 MiB respectively at M32768, and double at M65536. This is component
allocation size, not proof a full service still retains the 1M KV pool.

## Numerical and determinism evidence

Both candidates change FP32 summation/FMA order, **not bit-exact to production**.

- Five sampled-input mutations/shape, raw selected rows checked against FP64 CPU
  dot, reversed-row tests and 100 graph replays passed in the timing screen.
- Separate validation streams all 86 original Fn matrices (43 target layers,
  attention and FFN) from checkpoint, with independent random BF16 activations.
  All outputs finite; max normalized-mix abs difference vs production is
  7.6294e-5 (split4), 3.0518e-5 (split16); worst relative L2 1.2712e-6 / 6.5072e-7.
- Sampled real layer0 activations have different scale and error: roughly
  max abs 4e-4 / 1.8e-4, relative L2 4.8e-7 / 1.7e-7. Do not substitute the
  random-input absolute-error bound for real service inputs.
- 14 M values: 1,17,63,64,65,127,129,8191,8192,8193,32767,32768,32769,65536.
  Ten random-Fn mutations per M/candidate. Random row permutation, first65-row
  cross-M equality, repeated graphs, and changed-Fn graph versus eager all exact
  **within each candidate**. M8192 ran 1000 replays; other shapes ran 100.
- Initial validation failed a fixture-count assertion before tests: it included
  6 MTP Fn tensors, yielding92 instead of86. Fixed to explicit `layers.` names.
  This was harness selection, not a GPU/model numerical failure.

No teacher-forced logits, long-output quality, eight-rank service stability or
service ABBA has been done for this candidate. Passing this bounded component
suite does not close the whole-model drift question.

## ISA / resource evidence

Actual compiled gfx90a code contains `v_mfma_f32_16x16x4f32` and
`global_load_dwordx4`. K64 padded split4/16: 62 VGPR,29 SGPR,17920 bytes LDS,
wave64,zero private scratch and zero register spills. Raw metadata/disassembly
is archived; actual hsaco SHA256 recorded in cooperative-analysis.json.

All GPU work used HIP_VISIBLE_DEVICES=5, measured PCI B3:00.0 / rocm-smi card7.
No TP8 service was running concurrently. Check processes before future trials.

## Next decision

Component gain is sufficient to justify a **default-off** service candidate.
Prefer split16 first for its lower numerical perturbation; retain split4 as the
faster tradeoff only if complete-model checks support it. Guard original V4,
TP8 native ordinary EXTEND, existing mix-pair context, M8192..65536 and shape,
dtype,contiguity,device contracts. AR/draft/verify/V4.1 must not enter.

Then compare actual live mixes/teacher-forced logits and repeated fresh-cache
real-code outputs, preserve France smoke, verify 1M pool and all-rank path hits,
and only then service ABBA. Keep the accepted exact path available. Do not add
the component percentage to the 8.96k E2E result.

Artifacts: `.agents/experiments/dsv4_premix_mfma_20260916/`. The initial scalar,
unpadded and padded sources/results are preserved separately, so later header
edits cannot silently rewrite the evidence for an earlier binary.
