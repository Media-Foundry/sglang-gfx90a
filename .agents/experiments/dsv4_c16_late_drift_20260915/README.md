# Denser same-input drift frontier: first observed at C4 layer24

Two fresh services use the accepted original-V4 TP8 C16 owner configuration
(7.95k checkpoint), original weights,1M logical KV and32K prefill budget.
Both send the same16 real code requests with fresh salts and one output token.
Input echoes16/16 and France pass per service; all owned processes stopped.
These synchronous diagnostic runs do NOT measure throughput.

Capture only the first large forward, M32767/cases0–3, at13 C4 checkpoints:
2,20,22,24,26,28,30,32,34,36,38,40,42. All8ranks are captured,104 records
per service. The hook is default-off; the new layer-list and full-fixture flags
only customize diagnostics. No production arithmetic or performance flags changed.

`analysis.json` checks raw sample/page-file SHA256, recomputes logical KV
using the corrected preshuffle decoder, and verifies identical input IDs,
positions, prefix/extend lengths, sampled row placement and request identities.

Results:

- Within each service, complete x/q_lora/Q/weight hashes and logical KV match
  across all eight ranks at every captured checkpoint.
- Across services, layers2/20/22 match on all8ranks for those tensors and KV.
- At layer24, full hashes differ on all8ranks; every subsequent captured
  checkpoint differs as well. Same input metadata remains proven.
- At layer24, ALL sampled x/q_lora/Q/weight differences are zero even though
  full hashes differ. The9-row sample misses affected rows. Do not present
  zero sampled errors as numerical equality or a bound on the actual error.

This locates the **first observed indexer-checkpoint difference**, NOT the first
bad model operator. Matching normalized input does not imply matching MHC
multi-stream residual/post/comb state; a latent difference could predate layer22
and become visible after rounding later. Do not claim that layers<22 are all
numerically identical, or that CK stage2 caused this instance without a probe.

Next: capture entry/deferred MHC residual state and attention/FFN boundaries
around20–24, including per-row identities/hashes or full values at the affected
rows. Existing stage-dump callbacks cover many operators, but their current
sample-only mode and missing deferred entry state need explicit closure.
Keep source/admission immutable across the comparison. Then replay the first
divergent operator on frozen identical inputs; only afterward choose a fix.

The capture tool saves original referenced packed pages and sampled values
locally for every rank. To bound the repository archive, full tensors for rank0
are bundled; other ranks' raw tensor files remain local and their hashes are
in records/manifests. No full-model weights or 270MiB full-Q fixture are uploaded.

`sweep.py` refuses output reuse and only starts B after A completes and its
owned process tree is stopped. `analyze.py` refuses analysis if inputs/row
metadata differ. No global drift resolution or new speed claim is made.
