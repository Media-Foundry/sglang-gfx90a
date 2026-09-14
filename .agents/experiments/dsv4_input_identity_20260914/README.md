# C16 input identity and shape-dependent collective drift

Date: 2026-09-14. Base: `9b56854996`, original DeepSeek V4 Flash,
TP8/EP1, native AR, 1,048,576 logical KV tokens, chunk 32768, fixed-slot CK OFF.
These are numerical diagnostics, **not performance acceptance measurements**.

## Input contract

`run.py` uses 16 distinct public-source code requests from the preceding C16
experiment. Every wave saves the complete outbound payload, supplies unique
request IDs and cache salts, requests the server's actual prompt IDs, matches
responses by request ID, and verifies every prompt token against the manifest.
Temperature, output length, EOS policy and logprob options are held constant.
Cached tokens must be zero. Full GPU input IDs and positions are also saved.

A order is cases 0..15. B swaps cases 2 (8191 tokens) and 14 (8192 tokens).
This is a deliberate batch-composition intervention, not a claim that the
entire batch is identical. Shared requests retain identical IDs and positions.
The last prefill group changes from M32768, cases 12/13/14/15, to M32767,
cases 12/13/2/15. Case 15 position 0 moves from row 24576 to 24575.

`analyze.py` reconstructs the GPU sequences and aligns sampled activations by
(case, absolute position), never by flattened batch row. Debug dumps sample
positions 0/511/2047/4095/8191 while retaining complete IDs/positions.
Only the final prefill group remains in each trace directory; the evidence
does not cover every row, layer, or decode step.

## Service observations

Rank-0 A1/B1/B2/A2: server echoes pass for all 16 requests in every wave.
A1/A2 outputs match 16/16; B1/B2 15/16; A1/B1 14/16 (128 output tokens).
Rank-0 sampled shared rows match through `wo_b_partial`; after the collective,
case 15 position 0 differs in 2447/4096 BF16 elements, maximum 0.03125.

The follow-up `all-ranks` A1/B1/A2 captures all eight ranks. Input and normalized
attention inputs match. Rank 1 has a tiny Q difference that disappears at
attention output. Ranks 5/6 have propagating `wo_a` differences (max 0.00024414
and 0.00003052), then `wo_b_partial` differences (max 0.001953125/0.0009765625).
Thus **collective rounding is not the only source**. Rank-0-only tracing would
incorrectly attribute the whole problem to communication.

## Independent causal collective oracle

`large_ar_oracle.py` freezes eight real A1 partial vectors. It places the SAME
vectors into otherwise zero BF16 [32768,4096] row 24576 and [32767,4096] row
24575. Input values are identical; only shape/offset change. All ranks' outputs,
repeatability, the zero background and FP64-sum/BF16-round reference are checked.

`large-ar.json` independently reproduces 2447 changed elements with BF16 RCCL.
FP32 RCCL and expanded-capacity AIter new/legacy have zero changed elements and
match the reference on this fixture. This establishes shape/offset-dependent
BF16 reduction as one causal contributor, not a hardware failure or a proof of
whole-model determinism. FP32 accumulation is not universally order-independent.

The existing AIter adapter defaults to a 64-MiB eligibility cap, whereas these
hidden tensors are about 256 MiB. Expanding capacity in the oracle permits the
AIter comparison. It does not establish a production hit without runtime proof.

## Grid screen

`large-ar-sweep-v2.json`: two shapes, 8 ranks, five measurements after warmup,
rank-max of per-rank medians. Includes staging/casts. Not ABBA and not E2E.

| Backend | M32768 ms | M32767 ms | Shifted-vector changes |
|---|---:|---:|---:|
| RCCL BF16 | 4.515 | 4.387 | 2447 |
| RCCL FP32 | 8.211 | 8.292 | 0 |
| AIter legacy 16 CTAs | 5.455 | 5.428 | 0 |
| AIter legacy 8 CTAs | 6.512 | 6.485 | 0 |
| AIter legacy 32 CTAs | 6.841 | 6.747 | 0 |
| AIter legacy 48/64/80 CTAs | 7.994–8.492 | 8.000–8.517 | 0 |

The initial sweep failed at an overly narrow CUDA-only TensorView device guard,
before executing the candidate kernel. The v2 wrapper accepts matching ROCm or
CUDA DLPack GPU types and validates contiguous tensors; v2 completed with exit 0.
PyTorch IPC-export shutdown warnings are present; all rank processes terminated.
The wrapper is an explicit debug oracle, not connected to production dispatch.

`large-ar-chunks.json` also tested 1024/2048/4096/8192-row staging chunks with
both AIter implementations. All preserve the frozen shifted vector, but the
best chunked legacy result is ~5.72 ms, worse than both unchunked legacy and
RCCL BF16. Smaller chunks add launch/staging cost; none is promoted.

## Service-level repair diagnostic

`SGLANG_DSV4_DEBUG_PREFILL_ATTN_AR_FP32=1` is a new default-off diagnostic.
It covers only original V4 eager native TP8/EP1 attention-output BF16
[M8192..36864,4096] reduction, with FP32 process-group sum and BF16 output.
Decode, draft, target verification, small prefill and expert reductions are
excluded. A 32768-row call temporarily requires 512 MiB for FP32 accumulation.
`test_contract.py` verifies the eligibility boundaries and sampled-dump inputs.
The `fp32-attn` experiment retains the same A/B input-identity contract and
all-rank stage tracing. Its result must be evaluated separately from the oracle;
this is not enabled globally or accepted as a performance optimization.

The fresh `fp32-attn` service completed warmup/A1/B1/A2, with all 16 server
input echoes exact and 1M KV confirmed in its startup log. All eight ranks logged
the actual FP32 selection. At the layer0/case15/position0 boundary, the output
now matches the FP64-sum/BF16 reference exactly for each batch separately. The
A/B difference drops from 2447 to **4 elements**, attributable to changed incoming
partials (rank5/6 wo_a). Maximum difference drops from 0.03125 to 0.015625;
FFN-input max difference drops from 0.005859375 to 0.0009765625.

This does **not** establish whole-model stability: full 128-token outputs match
14/16 for A1/A2 and 14/16 for A1/B1; the corresponding all-rank control run was
15/16 and 14/16. Later-layer/FFN/decode differences remain. Four inspected output
prefixes are coherent source-code analyses, not a semantic correctness oracle.
Service debug timings are not compared as a performance benchmark.

## wo_a isolation

`woa_oracle.py` reads the real layer0 raw FP8/E8M0 wo_a tensors and applies the
same block128 dequantization and TP8 shard as the loader. For ranks5/6 the
M32768 einsum exactly reproduces the saved service output. Moving the SAME input
vector to M32767/row24575 changes one output element for each rank.

`woa-library.json` (GPU4, seven repeats, component only): M padding to 128 does
not fix this, despite making the matrix shape identical. F.linear, contiguous
transposed weights and disabling BF16 reduced-precision reduction also retain
the one-element drift. Computing `weight @ x.T` and transposing output removes
this fixture's drift but costs ~3.97–4.44ms versus ~2.03–2.12ms baseline.
Existing fixed-K64 Triton also removes drift but costs ~5.23–5.29ms, and differs
from baseline by one element on rank5. No library variant promoted.

Two offline fixed-order tile screens (`woa-tiles*.json`) reduce that cost to
**2.412–2.462ms** with BM128/BN128/BK128, 8 waves, 2 stages. The baseline in
the same screen is ~2.02–2.11ms. `woa-mutation.json` tests this tile with all
20 sampled real input rows, BF16 amplitude perturbations and row displacements
1/2/63/64/65/127/128/129: **100/100 shifted-exact trials on each of ranks5/6**.
Maximum BF16-output difference from a FP64 dot reference is 0.001953125.
This is component validation, not bitwise equivalence to the baseline or E2E
acceptance; the candidate remains only in `woa_tiles.py`, not service dispatch.

One larger tile requested 81920 bytes LDS, beyond the 65536-byte limit; the
initial second screen aborted before launching it. The completed v2 records
these resource rejections explicitly and continues other candidates. No such
failure is counted as a successful measurement.

## Scope and outstanding work

- Input mismatch is excluded for the shared requests in these experiments.
- Batch grouping/row movement changes real projection and reduction numerics.
- A slower stable collective is not promoted as a speed improvement.
- Further work: combine the validated row-stable `wo_a` candidate with the
  opt-in FP32 attention collective, check the next first divergence, and run
  full prefill-to-decode correctness before any performance acceptance.
- Earlier CK fixed-slot fix remains default OFF; it addresses a separate atomic
  reduction contribution and does not solve the current batch-dependent drift.

## Artifacts

Local traces contain model weight dumps; **do not publish them**. The evidence
archive excludes all trace directories and keeps request/response evidence,
summaries, commands and logs. `summary.json` records local trace file hashes.
See `package_evidence.py` for the explicit packaging policy. Live requests and
the diagnostic services were owned and stopped by the harness; AMD-SMI process
checks precede each eight-GCD experiment.

Example controlled service commands (DS conda environment):

```bash
python .agents/experiments/dsv4_input_identity_20260914/run.py \
  --run-name new-control --rank -1 --short
python .agents/experiments/dsv4_input_identity_20260914/run.py \
  --run-name new-fp32 --rank -1 --short --fp32-attn-ar
python .agents/experiments/dsv4_input_identity_20260914/all_ranks.py \
  --run-name new-fp32
```

Each run name must be new. `--baseline-dir` can point at an extracted directory
containing the baseline `inputs.json` and `start-ar-matrix.sh`. The harness
explicitly sets FP32 diagnostic 0 or 1 and CK fixed-slot 0, avoiding inherited
arm selection. It uses the existing owned-process lifecycle and stops its own
service in `finally`; it does not terminate unrelated GPU workloads.
