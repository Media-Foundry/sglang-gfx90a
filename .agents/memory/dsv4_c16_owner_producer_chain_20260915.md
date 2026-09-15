# Owner producer: complete-chain row-stability gate (2026-09-15)

Previous goal turn was progress: live-object producer speed opportunity and
row/shape-dependent BLAS differences were established. This turn extends that
evidence to the complete Q/weights/score/Top-K chain; accepted production
performance remains7953.956103 input tok/s. No launcher or production math
change, no new E2E speed claim. Original V4/TP8/EP1/native AR,32K chunk,1M KV.

## Input identity really checked

Two new fresh diagnostic services: `capture-rocblas` and
`capture-rocblas-fullweights`, under
`.agents/experiments/dsv4_c16_owner_producer_20260915/`.
Both completed16 exact request input echoes, zero cache hits, one output token,
France/Paris check, source-freeze assertion and owned shutdown.

Against the earlier default `capture`, ALL THREE have identical first-forward
M32767, input IDs SHA, positions SHA, full layer20 x SHA and q_lora SHA:

- IDs:bf9f00b108d29f2fdd1e89edcc3dc563c82577a1a61de3f96be6f28f6410d4df
- positions:630b2cc378ddcc92993941a7647c13d9bfcdec8d8e8f2e5c486ab37c2d4b1e59
- x:b6a9d0d6af863beb08c0d18d4473edbdcf6e186906fad92a09bc83a863368ee9
- q_lora:9f0708218ce880123ed15c03caf9f672609e66ef5727adf08d846238aae41ca1

Each recomputed original full producer matches that process's preexisting Q
and folded weights byte-for-byte. This localizes the observed candidate
difference independently of admission, input tokens or earlier-layer drift
for this fixture. It does NOT prove every full-model run is deterministic.

## Full compaction with rocBLAS: Q exact, small weights projection still differs

All eight owner plans have3072 valid rows (24576 total). With global BLAS set
to rocBLAS ONLY inside a synchronized rank0 diagnostic, full and compact raw
wq_b and FP8 Q match exactly. Each compact producer and reverse-row test is
repeatable. Other ranks wait; production resumes only after BLAS restoration
in a finally block. No candidate result is consumed by the model.

The small64-output `weights_proj` still differs with M:8..21 changed raw bytes
per owner,23..59 changed folded-weight bytes, score max_abs0.00365..0.01103.
Top-K discrepancy against same-backend full producer:

- owner4:2 rows,207 positional ID changes,2 removed/2 added KV entries.
- owner6:1 row,206 positional changes,1 removed/1 added entry.
- other six owners:zero selection differences.

So413 positional differences mean just THREE actual membership replacements,
not413 separately selected KV changes. Raw full/compact/production logical ID
tensors now saved locally, making membership counts independently auditable.
`selection_delta` ignores negative padding and counts set differences only on
affected rows; CPU test explicitly distinguishes reorder from membership.

## Narrow remedy: leave small weights projection full-M, compact only Q

The second fresh service computes `original_initial_weights` before changing
the diagnostic BLAS preference. Both references and candidate use these same
original full-M weights, gathering corresponding rows for the compact path.
Only wq_b and subsequent RoPE/Hadamard/quant use the owner subset.

For all eight owners, versus the same rocBLAS full-Q reference with ORIGINAL
weights projection, the following are byte-exact:

- FP8 Q and folded weights;
- complete scores;
- logical and physical Top-K IDs, membership and order;
- reversed-row Q/weights, after undoing permutation.

This closes the shape/row compaction discrepancy on the tested real layer20
fixture. It is NOT equality with current hipBLASLt production wq_b. Compared
to original production,57 of24576 active rows change membership,60 entries
removed and60 added. No selected KV is arbitrarily dropped: the differences
come from changed floating arithmetic and must pass the user-allowed-small-
drift E2E/logit gates before any rollout.

Live-object full-vs-fully-compact ABBA medians7.02036/0.78829ms. Additional
full-vs-full-weights/compact-Q ABBA medians7.02228/1.05087ms. Three cycles,
five calls/event sample. IMPORTANT: the latter timed full weights GEMM runs
under probe rocBLAS, while correctness uses saved original weights results.
Thus it is a cost screen, not timing of the eventual exact per-op production
integration. Eight-rank service critical-path gain remains unmeasured.

## Explicit per-operation rocBLAS ABI prototype

`explicit_rocblas.py` is experiment-local; NOT imported by production.
It owns a rocBLAS handle, sets the actual current HIP stream and calls
rocblas_gemm_ex with BF16 A/B/output and FP32 accumulation. Signature/enums
checked against installed rocBLAS headers. It never changes global BLAS
preference in its callable. Loaded library resolves to
`/opt/rocm/core-10.0/lib/librocblas.so.5.6`.

After services stopped and all eight GPUs were checked free, physical GPU4
only ran the ABI test with actual frozen input and SHA-verified BF16 weight:

- M3/17/3072/32767 exact against PyTorch rocBLAS;
- all four shapes also exact on an alternate, explicitly ordered stream;
-100 same-input HIP Graph replays exact;
-20 real-row order/amplitude mutations at fixed captured pointers exact.

This establishes the local call's ABI and ordered-stream/graph behavior, not
concurrent shared-handle safety. Eventual integration needs per-runner/stream
ownership and strict native-prefill/owner guards. Do not transplant the
process-global preference switch into production. Test reference selection
does temporarily set/restore it, whereas the explicit candidate never does.

## Validation and next step

Nine CPU tests pass; Python syntax and diff checks pass. Existing pytest
asyncio_mode warning remains. Both services stop cleanly. Candidate query
results were discarded, so France checks validate unaffected production,
NOT the candidate's full-model answer quality.

Next real task: guarded full-weights/compact-Q integration using explicit
per-operation rocBLAS, with complete metadata/fallback ownership. Keep complete
compressor/indexer KV updates and original logical KV capacity. Verify more
layers/irregular M, fixed-prefix target logits and fresh long answers; only
then service ABBA. A compact Q tensor cannot be passed into existing full-M
metadata trimming as if its row count were full M. AR/draft/V4.1 must remain
outside the selector. Existing CK stage2 atomic drift is separate and remains
unresolved at whole-model level.

Evidence archive `owner-producer-chain-evidence.tar.gz`:56403595 bytes,
SHA256 `819e01450a2bb2176066c50904912d8e3b9f6abb3d5890b00cc5a6ca1434d43b`.
Includes both fresh-service manifests, actual input echoes, logs/source patches,
oracle reports and complete per-owner logical-ID tensors, plus the explicit
ABI/graph test. Excludes only the large full producer `oracle/inputs.pt` files,
whose hashes are recorded in the reports and which remain local. Final GPU
check reports no running processes on all eight GCDs.
