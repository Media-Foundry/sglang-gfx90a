# V4.1 ROCm indexer cutoff stability and fitting checkpoint

Date: 2026-09-13. Parent commit: `91c485d335`.

## Scope and outcome

Original DeepSeek-V4.1-Flash weights, TP8/EP1/no A2A, eager execution,
Engram embedding tables in pinned host memory. Correctness-first bring-up,
not a performance checkpoint. Actual test pool is 8192 tokens, chunk 2304;
the model's advertised 1048576 context is NOT validated capacity.

The earlier repeat-prefill divergence is reproduced and repaired: ordinary
ROCm `torch.topk(sorted=False)` changes cutoff-tied membership on identical
small-M inputs. Sorting the selected IDs afterward does not fix membership.
`stable_index_topk` orders scores stably, resolves ties by logical ID, then
emits ascending logical IDs. Scores, weights and visible-key coverage are
unchanged. This is the V4.1 eager path, not the optimized V4 Top-K path.

## Evidence

Committed reports: `../experiments/dsv41_indexer_stability_20260913/`.

- Before: three identical 2560-token prefixes, each split 2304+256, rank0
  bounded traces. All 202 recorded first-chunk comparisons per repeat agree.
  Second chunk first diverges at layer2 MQALayer logical selection/output;
  191/202 comparisons differ per repeat. Total 382/808 differ.
- After: same input and split, 164 instrumented modules, all 808 full-byte
  output/logical-index hash comparisons agree. Physical page IDs are not an
  equality requirement. Hook samples are only samples; full hashes cover all
  output bytes. Hooks synchronize and are unsuitable for performance timing.
- Component: M256/N1280 and M72/N1188 each show valid-member differences in
  30/30 old-path repeats (maximum 38 and 13 changed rows); M2304/N1152 shows
  0/30. Stable path matches CPU reference on all three shapes, including empty
  and short causal rows. Each shape passes 1000 graph replays plus ten changed
  score replays. Report excludes invalid -inf filler IDs from old differences.
- Fresh post-fix cached baseline: 192 continuation tokens from a real
  2376-token source-review prompt. Full recompute agrees on next-token ID at
  all nine checked lengths: 2376/2377/2378/2431/2432/2433/2559/2560/2561.
  Logprobs are NOT identical between cached and recomputed paths: maximum
  common top-20 logprob delta reaches 1.62325. This is top1 agreement at the
  tested positions, not full numerical equivalence.
- Two independent fresh-cache 192-token generations agree in every output
  ID, every token-logprob row and every top-20-logprob row. Full-tensor trace
  copies were already capped/exhausted during these generations. They end by
  length, not EOS; the partial answer is coherent, not a complete quality eval.
- Four concurrent short requests (France/arithmetic/Python/SQL), two rounds:
  8/8 pass and per-case IDs repeat. France separately returns
  `The capital of France is Paris.`, natural EOS, zero cached tokens, 1.714s.
- Unit suite: bringup helpers, V4.1 OpenAI encoding, V4.1 DSML detector:
  40 tests plus 16 subtests pass. This is not a full V4 performance regression.

The after-trace probe originally compared against a PRE-fix cached baseline
and returned failure because token 343 differs from old token 362. That probe
is only within-configuration repeat evidence. The nine-position comparison
above uses a NEW same-version cached baseline, avoiding that confound.

## Reproduction and runtime

`scripts/rocm/check_dsv41_index_topk.py --output <new.json>` runs the isolated
GPU oracle. Check `amd-smi process` first. The initial isolated probe ran out
of free GPU memory while the resident service retained allocator cache; the
service did not crash. An idle-service `/flush_cache` released enough memory,
then the probe passed. Do not relabel that as a server OOM or kernel fault.

`scripts/rocm/dsv41_trace_hooks.py` now supports bounded summary-only traces,
rank selection, minimum row count and optional parameter capture;
`scripts/rocm/compare_dsv41_trace.py` compares full-byte hashes and explicitly
labels sampled errors. Its zero exit code indicates coverage, NOT equality;
inspect `full_hash_equal` fields.

After-fix live service at recording time: parent81910, tmux
`dsv41-prefixstable-20260913`, log `/tmp/dsv41-prefixstable-20260913.log`,
port30101, ready 11:51:04 HKT. Raw traces remain under
`/tmp/dsv41-prefixtrace-20260913` and `/tmp/dsv41-prefixstable-20260913`.

## Checkpoint dependencies

This checkpoint also retains previously tested but uncommitted V4.1 fitting
source: ratio1/2 HIP cache metadata and attention, FP32 ratio2 compressor
projection, tiny HC GPU placement (large Engram remains host), actual FP8
K32 scale handling, A16W4 shape admission, server feature validation and
default-off diagnostics. Without these dependencies a clean checkout would
not reproduce the tested V4.1 path. Unrelated experiments and the generated
CUDA graph memory pickle are deliberately excluded.

## Remaining correctness work

- HIP currently does not apply layer20 candidate-block prefilter to later
  index sources. Actual config is 2048 blocks of 8, so this matters above
  16384 positions; the current <=8192 pool does not exercise pruning. Implement
  and independently validate it before claiming longer-context support.
- Cached/recompute numerical differences are not fully attributed; independent
  full-model reference comparison and broad/long answer quality remain open.
- Larger concurrency, graphs, multimodal and 1M capacity are unvalidated.
- No speed gain claimed for stable sorting. Do not re-enable nondeterministic
  cutoff membership to obtain throughput.
