# Reset capture-produced metadata before alternate warmup

Source audit identified a concrete missing boundary in the paired diagnostic:
DSV4 `init_forward_metadata_in_graph` upgrades Raw -> Full only when metadata
is Raw. During capture this changes the Python object but records, rather than
executes, GPU metadata initialization. Calling the same forward closure eagerly
again without resetting therefore skips that initialization and consumes the
capture-produced Full object.

Both DSV4 attention implementations already provide `on_after_cuda_graph_warmup`
which restores `_current_capture_raw`. The normal runner uses it between warmup
and capture; the paired helper omitted it between baseline capture and candidate
warmup. Call that existing hook after draining baseline capture and before
alternate warmup. Missing hook now rejects the diagnostic. No changes to the
normal runner's metadata sequence, mathematical kernels, or KV sizing.

Ten real-method CPU pair tests and seven transaction tests pass. Added a
Raw/Full-recorded-not-executed state-machine regression and missing-hook failure
test. These establish ordering in the helper, not GPU fault resolution.

Full-model validation launched from validated baseline PID3507653 via the exact
clone utility, with pre-stop amd-smi ownership check and mode0600 private launch
snapshot. New startup state:
`/tmp/dsv4_tp8_paired_metadata_reset_20260908.json`, controller47984.
Service log uses `.service.log`; inspect state for PID. At this note startup is
pending, not correctness-validated. Original TP8/EP1 native1M KV config retained.
Do not declare the illegal-address root cause experimentally confirmed until
this startup and both-arm replay tests pass.
