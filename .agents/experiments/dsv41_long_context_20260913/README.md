# V4.1 bounded score workspace and long-context evidence

Component reports use isolated GPU4 with original synthetic FP4-grid Q/K,
not a model checkpoint run. `workspace-guard.json` is the final serving
selector screen, including early small-key prefill. `initial-long-only-guard.json`
is the first rejected guard which still bypassed small-key large workspace.
`prototype.json` retains the initial prototype; `64k-component.json` and
`ragged-component.json` extend numerical/selection coverage, not full-model
capacity. These are not ABBA performance results.

`initial-oom.txt` and `failed-long-request.json` record the failed first E2E
attempt; do not omit it when interpreting the guard change.

`prepared-input.json.gz` preserves the full real-source prompt, source hashes,
and22316 exact token IDs used for both failed and successful attempts.
Its uncompressed SHA256 is
`d588c2b1bd8802b66d22fdef29c946383fbcc3ab6245f8f77fbd1e792ac806b2`.

`long-response.json.gz` preserves the successful full HTTP response, request,
90completion IDs and all returned logprobs. Its uncompressed SHA256 is
`c6739a3f621a380c969ec73505d136764e4276f241dc6ec0971b83f2a6c96086`.
All seven requested facts agree; natural EOS, zero cached prompt tokens.
The prompt was frozen before the final guard edit; none of its queried facts
changed. Do not regenerate input when replaying this trial.

To inspect compressed JSON, use `gzip -cd <artifact> | jq ...`.
To replay, decompress the prepared input to a new local JSON path, then run
`scripts/rocm/check_dsv41_long_code.py --prepared <path> --output <new.json>`.
Check `amd-smi process` and service health before another GPU/request run.
