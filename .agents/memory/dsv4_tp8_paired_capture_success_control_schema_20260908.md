# Metadata reset restores full-model paired capture; control wire fix

Service3516787 successfully captured both graph arms after the raw-metadata
reset. All8ranks report extra_device_bytes=37748736 (36MiB) for the candidate
capture, overall graph0.68GB, available15.55--15.61GB, capture12.49--12.64s.
Original1M KV remains. The earlier warmup illegal-address failure did not recur.
This validates the missing reset boundary for this startup, not full replay.

Startup controller47984 exited0. Same-process ABBA controller68064 then exited1
before creating any measurement block: `/set_internal_state` returned `[False]`.
HTTP200 was initially mistaken for successful switching in commentary and was
explicitly corrected. A later idle retry also returnedFalse. No candidate
replay or new correctness/performance result from these requests.

Root cause of rejection: `SetInternalStateReq.server_args` is
`Dict[str, Union[int,float]]`. The actual Pydantic TypeAdapter converts JSON
booleanTrue into integer1; the new worker's `type(value) is bool` guard rejected
it before checking idle state. Reproduced with the real schema in DS Python.

Fix: preserve the shared request schema, accept only the sole integer0/1 arm
key, normalize internally to bool, and have the harness send integer0/1.
Eleven CPU pair tests pass including invalid/mixed numeric controls. Device
selection and all-rank idle/memory protection are unchanged.

New serial ABBA harness logs subprocess output to files, verifies C1 full IDs
against the fixed reference in every block, tests separate C32France, and uses
the same real-code C32 corpus for all arms. Final success restores baseline arm.
It never restarts/kills services and fails rather than bypassing a rejected arm.
Raw C32 output hashes still require final review.

At writing, service3516787 is alive on baseline arm; it has the old control
guard loaded. Restart explicitly with the fixed source before rerunning ABBA.
Failed state `/tmp/dsv4_tp8_paired_same_process_abba_20260908.json` is retained;
use a fresh state path. Capture log is the metadata-reset `.service.log`.
