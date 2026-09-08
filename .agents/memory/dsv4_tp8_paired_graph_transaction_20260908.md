# TP8 same-process down-uniform comparison: capture transaction

## Motivation and scope

The two restart-based ABBA runs showed about +0.55--0.64% resident C32
throughput with down-uniform enabled. C1 was slower in those candidate processes,
but the restored baseline process also measured 83.404895 tok/s without the
flag. Kernel causality is therefore unresolved. Preserve the original C1 graph
and compare two M32 graphs in one process before deciding on promotion.

This step adds only an **unconnected experimental primitive**, not a production
selector, additional service graph, or new accepted performance result. The
native TP8 baseline PID 3468727 was confirmed live; no service was restarted and
no GPU experiment was run. The 1,048,576-token pool configuration is unchanged.

## Implemented contract

`runner_backend/paired_graph_transaction.py` captures a candidate through an
injected existing capture routine into a distinct pool. It retains the graph,
output object and pool as one unit, and always restores the baseline graph,
output mapping, backend pool and allocator pool. Missing baseline, no-op
capture, output-object reuse and shared-pool use fail loudly. This does not
prove device-memory disjointness; the earlier eight-rank dual-pool oracle tested
the runtime pool/IPC prerequisite separately.

CPU tests: `DS/bin/python scripts/rocm/check_dsv4_paired_graph_transaction.py`.
Seven tests passed, including failures before graph installation, between graph
and output installation, after both installations, and allocator setup failure.
C1 graph and output identities remain unchanged in these tests.

## Control delivery audit

Tokenizer `set_internal_state` sends through its communicator. Scheduler
`SchedulerRequestReceiver._broadcast_reqs_across_ranks` broadcasts control
requests on the TP CPU group (or all requests on that group without DP
attention), and `process_input_requests` dispatches them on every worker.
This supports an all-rank idle consensus for the intended TP8/EP1, no-DP/CP/PP
profile. It is not permission to enable a general topology-independent control.

## Remaining before service testing

1. Wire capture-only down-uniform override, exact native TP8 M32 guards, and
   backend graph/output selection. Other tiers, especially C1, stay baseline.
2. Capture inside the existing communicator capture/IPC-registration scope,
   preserving warmup and attention reset hooks.
3. Add idle-only all-rank arm switching, with strict control request validation.
4. Measure full-device additional memory, not just Torch allocated bytes;
   enforce a predeclared cap and retain the full 1M KV pool.
5. Fresh-process correctness, then same-process ABBA with fixed real diverse
   code inputs; verify C1 and C32 outputs after each arm change.

Do not describe this CPU transaction test as full-model graph or numerical
correctness validation. No default performance flag has changed.
