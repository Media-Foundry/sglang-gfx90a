# CK helper committed-source contract repair (2026-09-08)

## Confirmed defect

Both reviewer snapshot `832604af` and pre-repair HEAD `28a0b44347` fail
`ast.parse` for `gfx90a_bf16_batched_moe.py`, line 146. The
`gfx90a_bf16_ck_moe(` declaration precedes the scale helper definition, and
`scales_shuffled` is outside the keyword-only argument list. Working-tree source
already contained the declaration repair, alongside unrelated FP16/loader
experiments. A working process or short decode benchmark cannot validate the
committed lazy-import CK path.

## Repair scope

Commit only the three existing declaration/interface repair hunks: define the
scale helper first; restore the CK function declaration; place
`scales_shuffled: bool = False` before the closing signature. No math, layout
mapping, kernel selector, workspace, or precision change. Preserve other dirty
edits without including them in this repair.

Added `scripts/rocm/check_dsv4_ck_helper_contract.py`. It uses only Python's
standard library, parses the complete helper, verifies helper ordering,
keyword-only boolean default (not tuple), no annotated shadow assignment, and
the production caller's keyword compatibility. `--revision` tests Git objects,
not the working tree.

## Verification and limitations

- Regression reproduced on pre-repair HEAD; working-tree check passes.
- The repaired staged Git tree is separately checked before committing.
- This is a syntax/API repair, not new prefill performance evidence.
- The reported 6420.39 input tok/s result remains TP8, P32, 131072-token pool;
  a clean-commit fresh-process large-M CK reproduction is still required. Do
  not relabel it as a 1M KV-pool acceptance result.
- Paused shared-gate performance screening to prioritize this issue. Its first
  screen completed around 83.53 C1 tok/s, but actual candidate dispatch was not
  established; this must not be claimed as either a fused-kernel win or loss.

Next prefill experiments should first close clean-source reproducibility, then
consider active-query production trimming while retaining every indexer cache
update. Query-row sharding remains an unmeasured candidate, not an accepted gain.
