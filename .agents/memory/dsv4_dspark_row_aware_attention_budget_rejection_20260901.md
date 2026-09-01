# DSpark M128 row-aware sparse-attention budget screen (2026-09-01)

## Scope

- Physical GPU 4 (`HIP_VISIBLE_DEVICES=4`).
- Original DeepSeek-V4-Flash weights and the current CK split-K=2 sparse decode kernel.
- Gamma-three M128 layout: every fourth row is an anchor; the other 96 rows are drafts.
- Anchor rows retain Top-K/context 512. Draft rows are capped independently.
- This is an oracle-only diagnostic. It does not alter the AR or service path.

## Correctness

Every case passed 100 input mutations within the existing CK-vs-Triton tolerance and
1000 HIP graph replays were bitwise stable.

## Results

| Draft context | Total indices | CK latency |
|---:|---:|---:|
| 512 | 65,536 | 182.079 us |
| 384 | 53,248 | 164.972 us |
| 256 | 40,960 | 175.769 us |

Draft context 384 saves 17.107 us/layer, or about 0.74 ms over 43 layers. Draft
context 256 is slower despite fewer indices because the fixed split-K work and load
balance become less favorable.

## Decision

Rejected as a primary optimization: the best measured saving is below the 5% E2E
checkpoint budget and changes the sparse-attention budget/semantics. Keep the
benchmark switch for future DSpark quality/performance studies, but do not enable it
by default and do not expose it to native AR.
