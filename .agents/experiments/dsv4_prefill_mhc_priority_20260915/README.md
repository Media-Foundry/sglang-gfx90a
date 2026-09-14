# Prepared large-prefill MHC priority oracle

Status: v3 completed on physical GPU4. Correctness checks against matched R8
passed, but the allocating full boundary is slower: M8192 3.0185->3.3209ms,
M32767 11.9294->13.2645ms, M32768 11.9655->13.2801ms. Do not promote the
priority change as a speed optimization. Existing multi-request pre-mix8
remains accepted; this screen compares a different, legacy single-request
full boundary. No production selector is changed.

Two harness failures are retained in `first-attempt.md`. The corrected harness
uses the existing single-GCD pattern (no TP group, symmetric allocation
disabled), restoring both accessors after every call. This is not a TP8
collective or mempool benchmark. Do not run concurrently with a service ABBA.

The32K service confirmed that request batch1 admits the old FP16-weight
split-K tail even at M32767. This screen compares the complete MHC boundary:

- A: existing batch1, launcher-equivalent FP16 split-K path.
- B: same inputs, suppress legacy admission only in active M8192..65536
  prefill; use the existing FP32 pre-mix8 path.
- R: existing batch2 FP32 dispatch, which uses20 Sinkhorn iterations.
- R8: R with only its Sinkhorn batch hint aligned to1, selecting8 iterations;
  the matched-iteration numerical reference for B.

M1 must preserve the legacy path exactly. Large M requires B/R8 byte equality,
finite outputs and row-permutation invariance. A/B equality is not expected:
the cached Fn dtype and reduction order differ. This is not a new kernel or
an E2E speed claim. Any service selector requires its own scope/correctness
tests and ABBA, with AR and speculative paths preserved.

The five layer0 rank0 captures are read from
`.agents/experiments/dsv4_input_identity_20260914/trace-B1/`:
`ffn_mhc_residual`, `hc_ffn_fn`, `hc_ffn_scale`, `hc_ffn_base`, `ffn_norm_weight`.
They are local large-tensor prerequisites, not bundled in this directory.
The residual is repeated/sliced into test shapes; post input is initially
zero and combine identity, then perturbed. These are synthetic boundary
fixtures based on real captures, not fresh32K model activations. File hashes
are recorded on execution.

The model-level sinkhorn argument must remain20 to select the gfx90a path.
The environment override8 applies only to batch1 in `hc_split_sinkhorn`;
batch2 otherwise uses20. Attempt v2 wrongly expected B/R bit equality despite
this difference and failed. R20 remains a diagnostic while R8 explicitly
aligns this one hint; A/B preserve their current8. Passing8 as the model
argument would incorrectly enter the unsupported fallback. Small M retains
the legacy batch1 path for every arm.

After the service has stopped and all GPUs are free:

```bash
HIP_VISIBLE_DEVICES=4 SGLANG_USE_AITER=1 \
PYTHONPATH=python:python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:python/sglang/kernels/aot/python \
/home/pc/anaconda3/envs/DS/bin/python \
  .agents/experiments/dsv4_prefill_mhc_priority_20260915/oracle.py \
  --sizes 1 8192 32767 32768 --mutations 10 \
  --output .agents/experiments/dsv4_prefill_mhc_priority_20260915/screen.json
```

The script checks GPU ownership before importing torch. Its temporary Python
admission override is restored in `finally`. Three component ABBA cycles
measure the allocating full boundary; output residual/post/comb/norm are all
checked. This does not benchmark TP8 collective or symmetric-pool behavior.

Each arm also has six same-input eager replays (`--replays`), compared against
a cloned snapshot to avoid a shared-workspace alias hiding drift. Candidate
replays must be exact; legacy replay differences are recorded separately from
cross-arm precision differences. Every A/B mutation checks finite outputs and
records differences, not only the initial fixture. These are component checks,
not proof of full-model repeatability or graph replay correctness.

`screen-small-v3.json` covers M1/M8192 with3 mutations each;
`screen-large-v3.json` covers M32767/M32768 with10 mutations each.
Both are complete. Every A/B replay check is6/6 exact, B/R8 and permutation
checks pass. The initial R8/R20 comparison differs only in the returned comb,
max_abs0.1672245562 for this repeated-residual/synthetic-post fixture. The
local model config requests20 iterations. This is evidence of a batch-size
math switch, not proof that it causes all service output drift.
