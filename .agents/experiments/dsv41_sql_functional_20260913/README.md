# V4.1 C4 SQL functional checks

`responses.tar.gz` contains the fixed prepared prompts/IDs, all8complete raw
HTTP responses (including selected/top20 logprobs),128fixture evaluations,
and the original summary. `summary.json` is also provided uncompressed.

All8generated queries pass their16independent fixtures. Completions contain
560-815tokens and end naturally. Output IDs differ between rounds for all
four cases. This is a functional smoke, not a published benchmark score,
performance measurement, general determinism proof, or validation of every
claim in the English explanations. See the memory note for the identified
dependency-closure explanation errors.

Scripts: `scripts/rocm/dsv41_sql_cases.py` and
`scripts/rocm/check_dsv41_sql_functional.py`. The fixture oracle uses Python
calculations independent of the generated query; model output is executed
only by an authorizer-restricted, in-memory SQLite connection.

The user requested a pause for an emergency job and explicitly asked that
the TP8 service be shut down. This was done and all8GCDs were verified free.
Do not replay this experiment or restart the service without a resume request.
