# C16 drift: complete KV history closes a sampling hole

Scope: original DeepSeek-V4-Flash, TP8/EP1, native AR, original checkpoint,
1M logical KV pool, 16 distinct real 8K code requests. This is a correctness
diagnostic, not a throughput measurement. Baseline source: `913d9108a9` plus
default-off prepare snapshots. No production selector was enabled by default.

## Evidence and limits

Experiment: `.agents/experiments/dsv4_input_identity_20260914/layer1-prepare/`.
The harness completed warmup/A1/B1/A2 and stopped its owned server. Every wave
verified all 16 full prompt-ID echoes, request IDs, zero cached tokens,
completion lengths, and decoded completion text. A1/A2 use the same request
order. B1 swaps cases 2 and 14; the last prefill changes from M32768 to M32767.
Case 15 keeps identical tokens and absolute positions but moves one batch row.

The diagnostic retains the previous fixed-order wo_a and FP32 attention/FFN
collectives. CK stage-2 fixed-slot is OFF. Snapshots are eager/synchronous;
their wall time and output-repeat counts must not be interpreted as production
performance or a proof of race freedom.

Earlier sampled closure of layer0 did **not** establish full-row closure.
Complete layer1 normalized inputs on rank0 reveal 147 changed rows of case15:
positions 512..639 plus 19 other positions. There are 255,936 changed BF16
elements, max absolute difference 0.001953125. Cases12/13 remain exact.

Layer1's complete QKV projection differs on those same 147 rows (102,692
elements, max 0.0078125). Actual KV after norm/RoPE differs on 147 rows on
**each of eight ranks**, with identical counts: 33,926 elements, max 0.0625.
The final 128 KV rows include 20 changed elements. Thus equal sampled Q at
the last query does not imply equal attention inputs: historical KV differs.

All retained complete input/QKV/KV comparisons A1/A2 are exact. Coverage is
only the final prefill group, not all 16 requests. Full 128-token outputs were
13/16 equal A1/A2 and 10/16 equal A1/B1; this is not whole-model determinism.

Layer1 has compression ratio 0: it has no indexer/compressor. The present
evidence does not implicate Top-K or a C4 cache boundary. Nor does it yet
implicate the layer1 attention kernel: its KV inputs already differ.

## Follow-up

`run_changed_rows.py` derives sample positions from the above complete tensor
comparison, then traces layer0 at all 147 changed positions plus six controls.
This avoids repeating the false inference from a sparse five-position sample.
The same diagnostic arithmetic and verified request-ID protocol are retained.
Further findings belong below only after the follow-up actually completes.

## Follow-up result: a QKV projection contaminates the sliding window

`layer0-changed-rows` completed all four waves and stopped its server. Complete
layer0 normalized inputs are equal for each of the three common requests.
The QKV projection changes four elements of case15, at absolute positions
512, 2048, 3328, and 3840. The actual normalized/rotated KV differs only at
position512 (six BF16 elements, max 0.0078125), identically on all eight ranks.
That KV remains visible to queries512..639 through the native128-token window.
This explains why a five-position sample missed a whole downstream span.

The sampled wq_b output also has independent small row-position differences;
fixing QKV alone cannot be advertised as a fix for all projections or all43
layers. All sampled A1/A2 layer0 stages are exact in this run, but complete
128-token outputs are only12/16 equal (A1/B1:11/16). Earlier prefill groups are
not covered by the retained traces.

### Frozen-input causal reproduction and repair candidate

`qkv_order_oracle.py` uses the two complete layer0 service inputs and the saved
real BF16 QKV weight, without reloading the model or changing checkpoint
precision. PyTorch `F.linear` reproduces **all elements** of both saved service
outputs exactly, including the four shifted-request differences. This rules
out the request wrapper as the cause of that projection discrepancy.

The fixed BM128/BN128/BK128, eight-wave MFMA tile gives zero differences between
corresponding complete request rows. All100 real-input perturbation/row-offset
trials are exact. The separately integrated service kernel matches the offline
candidate and also passes100 trials. The candidate does not equal the old
GEMM everywhere: on full outputs the differences are4355/4517 elements, max
0.001953125, because accumulation order changes. Do not describe this as
bitwise parity with production, only as row-position invariance on the fixture.

One-device component median: library3.1468ms, fixed tile3.8892ms; the other
tested tiles are slower. This is a correctness candidate, **not a speed win**.
`SGLANG_DSV4_DEBUG_PREFILL_QKV_STABLE` is default-off and guarded to original
V4/native/TP8/EP1/gfx90a/M8192..36864, normal prepare only. Decode and draft do
not enter it. Six CPU contract tests pass, including the actual callsite.
The service verification is recorded separately after completion.

`qkv_tie_analysis.py` computes CPU FP64 dots at the four discrepant coordinates.
All four lie close to a BF16 rounding midpoint: offsets approximately
1.94e-8, -8.22e-10, -3.78e-9, and -4.38e-9. In particular position512/column1478
is0.201660175634 in FP64, between service values0.2021484375 and0.201171875.
The result supports reduction/rounding sensitivity, not bad prompt IDs.
The A1 value agrees with the rounded FP64 reference for two coordinates and
the B1 value for the other two; neither service shape is a universal oracle.

### Service repair validation (completed)

`layer0-stable-qkv` completed warmup/A1/B1/A2, verified all16 input echoes each
wave, and stopped its server. All eight ranks logged the stable QKV hit.
Complete corresponding QKV on rank0 and complete KV on all eight ranks are
now exact across A1/B1; same-order A1/A2 complete inputs/QKV/KV also match.

Among the153 selected positions of case15, layer0 replicated attention-output
differences fell from141 rows to13 (max0.0625 to0.03125), FFN-output differences
from147 to16 (max0.015625 to0.0078125). These counts are **not** a whole-model
or whole-prompt accuracy percentage. The continuous512..639 contamination is
removed except for the independent query-side difference at512. Sampled
`wq_b` output still changes at some row offsets, with equal QKV producer input;
other projection/expert effects remain candidates for subsequent isolation.

All sampled same-order A1/A2 layer0 stages match in the candidate run. Complete
128-token outputs match13/16 A1/A2 and13/16 A1/B1. Because the control itself
is not fully repeatable, this must not be reported as a statistically proven
whole-output stability gain. Four inspected answer openings (cases0/3/10/15)
are readable code-review responses; this is not a comprehensive quality test.

`qkv_shape_row_axes.py` separates row offset from M: the same four real input
vectors placed in adjacent rows of **one** M32768 or M32767 matrix still show
the one-element differences. Changing M while keeping the row offset fixed
does not change those outputs. Thus row placement alone is sufficient on this
fixture; merely padding M to a graph/shape tier is not a fix.

Status: a real, service-validated **partial correctness repair**, default-off.
No production performance default changes; no new prefill speed claim. The
remaining work is to isolate `wq_b` and later boundaries, and to capture earlier
prefill groups for the remaining same-order whole-output drift. Do not start
by blaming layer1 attention or indexer: their inputs were previously unequal.

## Archived evidence and checks

`prepare-evidence.tar.gz` contains95 bounded evidence files (11,943,845 bytes),
SHA256 `31a842273d40549784fb00ad150cf5a4cf562497d267f8102558b7ae039d814e`.
The companion manifest records local tensor sizes and hashes; full activations
and model weights are not bundled. `validate_prepare.py` verifies the reported
fixture properties without requiring a GPU. Six CPU tests and source compile
checks passed. All owned diagnostic services and isolated GPU probes exited.

## Reproduction

```bash
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_input_identity_20260914/prepare_analysis.py \
  --run-name layer1-prepare --layer 1
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_input_identity_20260914/all_ranks.py \
  --run-name layer1-prepare --layer 1
```

`dsv4_prepare_dump.py` adds only default-off snapshots. Five CPU contract tests
passed, including disabled/no-GPU execution and full versus sampled storage.
Raw full activation tensors remain local; do not upload checkpoint tensors.
