# Fixed-prefix evidence after explicit resume

`france.json`: new-process France smoke, natural EOS.
`events-prepared.json`: fixed203input IDs from the previous event-session SQL
task, with original prepared-artifact SHA256.
`cross-batch-summary.json` and `cross-batch-recomputes.tar.gz`:24fresh-cache
recomputations against the old C4 generated766-token baseline. All actual
next-token IDs match; floating logprobs do not.

The previous full baseline is retained as`round0-event_sessions.json` inside
`../dsv41_sql_functional_20260913/responses.tar.gz`. Its uncompressed SHA256 is
`ad4faa139f4051db8c2efa2fe9851a1186ec0e07a203c92e376718e7e4d917f4`.

See`../../memory/dsv41_resume_prefix_20260913.md` for the corrected committed-ID
gate, limitations, runtime/storage provenance and completed fresh C1 control.
No model or kernel arithmetic was changed by this checker fix.

`c1-recomputes.tar.gz` includes the fresh738-token C1 baseline and all24fixed
prefix responses. `c1-summary.json` reports24/24actual-ID agreement and full
coverage; `c1-functional.json` confirms16/16SQL fixtures. Each fixed prefix
repeats exactly, but cached-decode versus full-recompute floating logprobs do
not. The archive is required to reproduce that distinction; summary success
does not imply whole-model bitwise equality.
