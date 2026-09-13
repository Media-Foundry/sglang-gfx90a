# Original DeepSeek V4 Flash: TP8 native AR revalidation

| C | P input tok/s | D resident output tok/s | D whole-wave HTTP tok/s |
|--:|--:|--:|--:|
| 1 | 4676.39 | 87.60 | 86.39 |
| 2 | 4989.25 | 109.94 | 102.41 |
| 4 | 5265.43 | 188.71 | 148.50 |
| 8 | 5099.23 | 334.18 | 254.45 |
| 16 | 5171.36 | 608.23 | 422.51 |
| 32 | 5254.98 | 1044.32 | 680.61 |
| 64 | 5250.05 | 1334.24 | 848.14 |

Long8K C32 resident ABBA speedup: 3.599x.

## Measurement scope

- P: retained completed baseline; exact new guard excludes prefill.
- D: new process, real512-token inputs, natural EOS, at most2048 output tokens.
- Each D round accumulates >=30s common resident windows. Three measured rounds; warmups excluded.
- Baseline vs final D is sequential matrix comparison, not per-tier ABBA. Outputs can differ.
- C1 uses the accepted separate GEMV-on three-round service supplement. Other-tier whole-wave HTTP rates retain their measured GEMV-off drain behavior.
- Long8K C32 comparison is separate A1/B1/B2/A2; do not apply its multiplier to short D.
- 1M logical token pool allocated; this is not a1M filled-context or accuracy test.
- P concurrency is client requests; admission16 and chunk36864 constrain actual GPU batch.
- Checkpoint precision unchanged; existing large-prefill BF16-CK is not bitwise SDOT.
- Fresh processes from the shared working tree, not a separate clean checkout; existing AIter/CK/local changes are recorded in environment.json.
- AMD VRAM samples ~5s apart, including startup/final matrix only; not exact allocator peak.
- Exact component score/TopK tests do not establish whole-model bitwise or factual correctness.
