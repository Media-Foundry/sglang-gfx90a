# Prepared large-prefill MHC priority oracle

Status: CPU syntax/interface checks passed; **not GPU-tested yet**. Do not
run concurrently with the active TP8 service ABBA. No production selector is
changed by these files.

The32K service confirmed that request batch1 admits the old FP16-weight
split-K tail even at M32767. This screen compares the complete MHC boundary:

- A: existing batch1, launcher-equivalent FP16 split-K path.
- B: same inputs, suppress legacy admission only in active M8192..65536
  prefill; use the existing FP32 pre-mix8 path.
- R: existing batch2 FP32 dispatch as a numerical reference for B.

M1 must preserve the legacy path exactly. Large M requires B/R byte equality,
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

The model-level sinkhorn argument must remain20 to select the gfx90a path;
both arms retain the current environment override8 internally. Passing8 as
the model argument would incorrectly enter the unsupported fallback.

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
checked. No result file exists at this preparation checkpoint.

Each arm also has six same-input eager replays (`--replays`), compared against
a cloned snapshot to avoid a shared-workspace alias hiding drift. Candidate
replays must be exact; legacy replay differences are recorded separately from
cross-arm precision differences. Every A/B mutation checks finite outputs and
records differences, not only the initial fixture. These are component checks,
not proof of full-model repeatability or graph replay correctness.
