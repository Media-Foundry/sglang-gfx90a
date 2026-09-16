# FP32 MFMA pre-mix screen: reject the scalar-loaded variants

Date: 2026-09-16. Base: `0a0b2e27f2`. Experiment only; no production code or launcher changes.

The accepted TP8 C16 prefill checkpoint remains **8964.910269 input tok/s**
(original V4, original weights, native AR, 1M logical KV, C16 x approximately 8K,
zero prefix hit). This turn did not run a new HTTP benchmark.

## Hypothesis and bounded implementation

Use native gfx90a `v_mfma_f32_16x16x4f32` to share each activation across sixteen
projection columns. Fn stays FP32; BF16 activations convert exactly to FP32 in
registers. No full FP32 activation expansion, no weight quantization, no atomics.

Four wave64s per CTA, each wave computes sixteen rows and sixteen columns. N=24
uses two column tiles. Screen serial K with load unroll 1/4/8, then K split
4/16/32 with fixed-order FP32 partial reduction. Include RMS finish and, where
applicable, all partial reduction in candidate timing.

This is not the historical full-FP32-activation `torch.mm` candidate. It changes
the summation/FMA order, so retaining FP32 Fn does **not** imply legacy bit identity.

## Observed component results

Graph ABBA, three cycles, five replays per timing sample, medians in milliseconds.
Input: repeated/scaled sampled real layer-0 tensors, not an independent live full-M
capture. Existing exact HIP post generates BF16 residual and RMS partials outside
the timed pre-mix interval.

| M | Production paired pre-mix | Best serial MFMA | Best split MFMA |
|---:|---:|---:|---:|
| 8192 | ~1.358 | 3.992 (unroll8) | 3.573 (split16) |
| 32767 | ~4.963 | 13.792 (unroll8) | 14.034 (split4) |

The serial M32767 candidate is 2.78x slower; split-K does not rescue large M.
M17 split32 is faster than this **large-prefill reference**, but M17 is outside
the target scope and was not compared against production small-M dispatch. Do not
claim a C1/decode optimization from it.

## Correctness and determinism

- Integer matrix mapping: five ragged shapes, including N24 and M65. Correct
  mapping plus unroll4/8 match FP64 CPU reference exactly. Deliberately alternate
  mapping fails, as expected; it is not a usable candidate.
- Every tested candidate/shape: five scaled-input mutations, finite output,
  byte-exact reversed-row remapping, byte-exact 100 graph replays.
- Legacy normalized mix is **not bit-exact**: serial large-M relative L2 around
  1.96e-6, max absolute up to about 0.00205; split16 large-M relative L2 around
  1.70e-7, max absolute about 0.000198 across tested shapes/mutations.
- Selected rows additionally compared raw dot products with FP64 CPU reference;
  raw errors are preserved in the JSON, not conflated with normalized mix error.
- No logits/teacher-forced/service correctness claim: candidates already fail
  component performance and are not integrated. This does not diagnose a new
  production whole-model drift source.

## Compiler evidence / interpretation

Actual gfx90a device object disassembly contains `v_mfma_f32_16x16x4f32`, with
scalar `global_load_ushort` activation and `global_load_dword` Fn supply. Split
variants report 32 VGPR, 18 SGPR, zero LDS/private scratch/spills, wave64. This is
not an observed register-spill failure. Scalar strided supply is a concrete
implementation limitation, but there is no stall-counter proof that it alone
explains the slowdown. Cooperative/vectorized supply remains untested; this
negative result does not reject all possible FP32 MFMA designs.

## Reproduction and artifacts

Directory: `.agents/experiments/dsv4_premix_mfma_20260916/`.
`mapping.py`; `screen.py` (serial); `screen.py --split`; `analyze.py`.
Use DS conda Python, ROCM_HOME/ROCM_PATH=/opt/rocm, /opt/rocm/bin first in PATH,
OMP_NUM_THREADS=1, HIP_VISIBLE_DEVICES=5. Scripts reject existing result JSON;
preserve evidence and use a fresh output location for reruns.

**Device-label correction:** HIP_VISIBLE_DEVICES=5 maps on this host to PCI
`0000:b3:00.0`, rocm-smi **card7**, not card5. The initial JSON's `physical_gcd=5`
was an ordinal-label mistake. Split screen records PCI explicitly. All GPU tests
were restricted by that same HIP mask; no TP8 service ran concurrently.

`initial-screen.tar.gz` preserves the exact initial sources and result JSON before
split support was added. Current sources/results plus device metadata and
disassembly are tracked separately. No weights, fixture tensor dumps, or compiled
binaries are committed. Production remains unchanged; move to a structurally
different supply/data-reuse experiment rather than extend the unroll/grid sweep.
