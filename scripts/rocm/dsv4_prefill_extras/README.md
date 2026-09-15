# Original-V4 gfx90a prefill extras: reproducible offline build

These are opt-in experimental helpers for the validated TP8 / C16x8K profile,
not universal SGLang defaults. They preserve checkpoint precision and the 1M
logical KV pool. H16 ownership is limited by its runtime selector; this does not
enable a new decode, draft, or target-verification path.

Build in the ROCm PyTorch environment, with an unused absolute output directory:

```bash
ROCM_HOME=/opt/rocm ROCM_PATH=/opt/rocm \
PATH=/opt/rocm/bin:$PATH \
/home/pc/anaconda3/envs/DS/bin/python scripts/rocm/build_dsv4_prefill_extras.py \
  --aiter-root /home/pc/pytorch/third_party/aiter \
  --rocm-root /opt/rocm \
  --output-dir /absolute/new/build-directory
```

The builder compiles packaged sources directly. It never reads an old
`build.ninja`, rewrites installed AIter/CK headers, or starts a GPU service.
`build-contract.json` pins the supported CK headers and compiler flags. Unknown
headers fail closed: audit their contracts before updating hashes. The two
private overlays select unique-assignment stores and guard invalid Set writes
explicitly (an offset-plus-2-GiB mask can otherwise wrap into a large live output).

Outputs include `build.json`, per-module manifests, compiler logs, dependency
files, and `profile.env`. Manifests retain compiler/PyTorch/HIP/ABI provenance,
non-system header hashes, commands, and shared-object hashes. Preserve the build
directory at its original location; runtime manifests contain absolute paths.
Failed builds are retained for diagnosis; retry into a new directory.

`source /absolute/new/build-directory/profile.env` selects the rebuilt unique-slot
CK and direct-HIP IPC modules **within an otherwise configured H16 prefill
profile**. It does not establish the complete launcher configuration by itself.
Use the recorded service launcher and verify all eight ranks' actual selections,
32768 prefill budget, and 1048576-token pool rather than just exported variables.

Numerical contract: unique-slot CK retains FP32 partials and fixed Top-6 reduction
order. It is byte-exact to the validated fixed-slot implementation, not a claim
of equivalence to unordered FP32 atomic accumulation. Scratch still includes
`[M,6,4096]` FP32 partials; this is not a memory-capacity optimization.

IPC buffers use direct HIP allocations. Consumers must finish using imported
views before those views are released and exporter allocations are freed. The
service uses runner/process lifetime; arbitrary hot replacement is not certified.

Validation evidence is recorded under
`.agents/experiments/dsv4_prefill_extras_build_20260916/` and the corresponding
memory note. Required checks include poisoned large-output store tests, distinct
sink peer tests, fresh-service attention comparisons, ABBA performance, and
repeated real-code continuations. No finite fixture establishes arbitrary-batch
whole-model bitwise invariance.
