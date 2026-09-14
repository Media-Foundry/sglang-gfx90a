# C16 prefill drift isolation — 2026-09-14

**Partial repair, not full dynamic-batch invariance.** Original V4 Flash,
TP8/EP1/no-A2A, 32768 chunk budget, allocated 1M KV pool, native AR only.
Base HEAD `e0aea46b77`; experimental implementation is archived with hashes.

## Findings

1. CK FP32 stage-2 atomic accumulation is genuinely nondeterministic for some
   fixed inputs. The frozen-input M8192 random-routing byte-level test records
   0/30 atomic replays exact, versus 30/30 for the unique-slot implementation.
   One BF16 output element can change even after the final cast. Regular large-M
   cases sometimes replay exactly on both paths: do not claim universal failure.
2. The much larger service-level output drift is strongly associated with batch
   composition/row placement. With the **same logprob options on both protocols**,
   one ordered batch-shaped HTTP request gives 16/16 repeated 128-token outputs
   in all three pairwise comparisons; 16 independent simultaneous HTTP requests
   give only 3/16. Saved `forward_entry_time` fields confirm different batch
   membership in the latter, not merely different completion order.
3. Fixed-slot stage-2 is not sufficient to repair that second problem. Its normal
   concurrent repeat score is 5/16, exactly the same count as the initial atomic
   service's normal concurrent test (different request permutations; not proof
   that the same five requests match). It also costs about 3.68% prefill throughput
   in a diagnostic A/B. **The new selector remains default off.**

## Component diagnosis and repair

The oracle freezes expanded BF16 W2, stage-1 intermediate, sorted IDs, sorted
weights, expert IDs and valid counts. It then reruns stage-2 alone. Separate full
stage repeats check the initialized sorter prefix and stage-1 output; these
remain exact in the tested fixtures. Synthetic weights use nonconstant E8M0
scales118--120; these tests are not a real-checkpoint layer oracle.

New optional selector:

```
SGLANG_DSV4_GFX90A_BF16_CK_FIXED_SLOT=1
```

It is consumed only inside the existing large-M BF16-CK FP32 stage-2 branch.
Default0 preserves the previous path. No CK binary or weight format is changed:

```
packed sorted ID (token, original slot)
  -> virtual token = token*6 + slot
  -> existing CK stage-2 with TopK=1 and [M*6,1,I] input view
  -> unique FP32 [M,6,4096] output (no competing experts at an address)
  -> fixed slot0+slot1+...+slot5 HIP sum, then one BF16 cast
```

Invalid/padded IDs map to the virtual-token sentinel; the original logical
router weights, stage-1 activation, B layout and K reduction are retained.
The current CK implementation fixes stage-2 KBatch=1. The Python helper validates
the materialized BF16/No-quant contract and the existing large-prefill shapes.
It does not silently fall back to atomics on allocation failure.

At M32768 the partial is **3 GiB/GCD**, versus the original **0.5 GiB/GCD** FP32
accumulator. It is temporary, not a persistent weight cache. Both real services
successfully retain the allocated 1M KV pool, but no filled-1M-context quality
claim follows from that allocation.

Representative component results (complete stage-2 including zero/reduction):

| Fixture | Atomic ms | Fixed-slot ms | Notes |
| --- | ---: | ---: | --- |
| M8192 balanced, initial Python sum | 2.574 | 8.144 | 0/20 vs20/20 elementwise replays |
| M8192 random, fused HIP sum | 2.829 | 4.328 | 0/30 vs30/30 elementwise replays |
| M32768 random, fused HIP sum | 10.413 | 16.326 | both30/30 exact in this run |

See the JSONs rather than mixing these different fixtures into one speedup.
`m8192-bitwise.json` additionally compares integer views of floats, passes100
graph replays, five changed-intermediate graph replays, and checks the integrated
helper against the frozen-stage oracle. Selected-row FP32 matvec references
agree within small floating-point reduction error; they are tolerance checks,
not exact old-path arithmetic. Older JSONs without `bitwise_checked` used
`torch.equal` on floats rather than integer views.

Full fixed-slot CK at M8192 and M32768 also passes row permutation and appending
one row while preserving the old rows byte-for-byte. This narrows the remaining
search but does not exonerate every CK shape or every real model input.

## Real service evidence

Same16 distinct public-source ~8K prompts; fresh salts and zero observed cached
tokens. Controlled waves generate128 tokens with `ignore_eos=True` and collect
output logprobs/top5. Three repeats per service. All saved ID lengths and local
tokenizer decoding match response text. France passes each fresh service.

| Service | Controlled output exact, three pairs | Controlled logprob exact | Ordinary concurrent output exact |
| --- | --- | --- | --- |
| Atomic | 16/16,16/16,16/16 | 15/16,16/16,15/16 | 5/16 |
| Fixed-slot | 16/16,16/16,16/16 | 16/16,16/16,16/16 | 5/16 |

The initial ordinary concurrent harness does not request logprobs. To remove
that confound, `service.py --mode atomic --matched-only` starts a fresh control
and requests logprobs on **both** ordered-batch and independent-request calls:

- Ordered batch: three pairwise full-output comparisons all16/16.
- Independent requests:3/16 full outputs exact.
- Ordered batch membership stays `[0..3], [4..7], [8..11], [12..15]`.
- Independent request memberships differ between waves; exact membership and
  first-split top5/margins are in `summary.json`.

The controlled atomic and fixed-slot services have **0/16 full128-token outputs
equal across paths**. Canonical slot reduction is not bitwise equivalence to a
particular atomic order. First splits occur at generated positions10--115;
some are ties, others are not. The candidate source-related prose shows no
obvious garbled/repetitive collapse in the inspected snippets. This is not an
executable-code quality score or proof that every numerical difference is benign.

Three ordinary P waves per service (actual131069 input tokens, one output token):
atomic median **5296.66 input tok/s**, fixed-slot **5101.91 input tok/s**, -3.68%.
This is a diagnostic A/B, **not** optimization-acceptance ABBA. Do not replace
the previous 5310.12 accepted C16 chunk-screen number with a mixed-protocol rate.

## What is and is not fixed

- Fixed: an opt-in mechanism eliminates competing FP32 atomic updates in the
  tested stage-2 contract; graph replay and integrated component checks pass.
- Improved diagnostic protocol: fixed-order batch submission provides a much
  stronger repeatability control than uncontrolled thread arrival plus final hash.
- **Not fixed:** arbitrary dynamic-batch/row-position invariance of model logits
  or long generation. Exact first offending operator in this8K/C16 workload has
  not yet been captured. Submission order and numerical execution must not be
  conflated with a proven scheduler bug or a complete correctness repair.

The historical row-stable prefill work already traced small-M differences through
Q/wo_a/compressor/router GEMMs and fixed its fixture. Its current selector is
default off and covers only M5..4096, so it does not protect the current32K rows.
Read `.agents/memory/dsv4_row_stable_prefill_diagnostic_20260908.md` before the
next round. Next meaningful work is large-M row/shape invariance and matched
teacher-forced layer boundaries, not repeating AIter sorting or BF16 CAS trials.
Do not simply widen that selector without a large-M numerical and cost oracle.

## Reproduction and evidence

- `stage2_oracle.py`: isolated physical GCD4; checks amd-smi before initialization.
- `service.py --mode atomic|fixed`: owned TP8 service, controlled and ordinary
  waves, one-token speed screen, PID/birth/command-checked cleanup in finally.
- `service.py --mode atomic --matched-only`: matched logprob protocol control.
- `analyze.py`: integer completion IDs, matching-prefix logprobs/first split,
  actual batch memberships; requires the matched control to finish.
- `pack.py`: hash-verifies all raw evidence and experimental source snapshots,
  refuses an existing archive, retains all original files, requires GPUs free.

No global production defaults, checkpoint weights, or small decode kernels were
changed. Services were stopped between arms. Keep the fixed-slot flag0 for the
fast profile; enabling it does **not** promise full E2E determinism.
