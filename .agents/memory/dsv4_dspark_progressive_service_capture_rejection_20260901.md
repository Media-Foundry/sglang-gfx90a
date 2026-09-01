# DSpark progressive M128 first service integration rejection (2026-09-01)

## Scope

After the exact four-rank oracle saved 120.344 us/layer, a strict temporary
production selector attempted to replace only the M128 target-verify all-reduce.
It was guarded by gfx90a + DSpark + TARGET_VERIFY + TP4/EP1 + BS32 + M128 and
was unreachable from native AR.

## What failed

The first launch failed loudly before capture because SGLang enters M128 target
capture without an eager M128 warmup; lazily allocating/registering peer buffers
inside capture is forbidden. Moving the per-layer input/output/sync allocation
and AIter registration to model construction passed that gate.

The second launch loaded the original target and bundled draft weights, verified
all three hash-router tables, and reached target graph capture. All four ranks
then segfaulted in `hipStreamEndCapture`/`CUDAGraph::capture_end`. This was not
an OOM and no inference request ran. The likely incompatibility is that the
standalone progressive primitive registered external meta buffers directly on
the AIter communicator while SGLang's full graph backend simultaneously owned
the communicator capture and graph-buffer registration lifecycle.

The service prototype was removed completely; `deepseek_v2.py` is back to its
pre-experiment contents. No broken selector or native-AR change remains.

## Continuation rule

Do not retry model-local `allocate_meta_buffer + register_buffer`. Production
integration must be owned by `GroupCoordinator`/`ca_comm` and participate in
the existing `ca_comm.capture()` plus graph-buffer registration lifecycle. A
safe design should expose the progressive operation as an out-of-place method
of the active communicator and obtain graph-stable registered input/output/sync
storage from that communicator's existing capture pool.

Before another full model load, make a minimal four-rank SGLang communicator
capture harness (not a standalone AIter communicator) and require 100 changing
inputs plus 1000 replays. Only then reattach the M128 model selector.

## Graph-pool retry

A second production attempt removed every external meta-buffer allocation and
manual `register_buffer` call. It allocated ordinary tensors inside the active
SGLang `ca_comm.capture()` context so that `get_graph_buffer_ipc_meta` and
`register_graph_buffers` owned the addresses. To isolate the test, only BS32
was captured.

This still produced a four-rank segfault in `hipStreamEndCapture`. Therefore
the failure is not explained by competing IPC registration lifecycles alone.
The remaining difference from the passing standalone oracle is composition
inside SGLang's full multi-stream target graph: the progressive begin and
anchor phases are split across streams already participating in other captured
events/collectives.

The model patch was again removed and `deepseek_v2.py` verified clean. The next
oracle must embed the progressive nodes in a minimal **full-backend-style**
multi-stream graph with the active communicator and reproduce SGLang's stream
joins. Do not pay for another model load until that capture-only harness passes.

## Existing-communicator-buffer retry

A third attempt reused the already registered `ca_comm.buffer` instead of
allocating any new graph storage. This was the first production composition to
finish both target and draft BS32 graph capture on all four ranks. It therefore
narrows the two earlier `hipStreamEndCapture` crashes to external/graph-pool
address ownership rather than the progressive kernels themselves.

The first real heterogeneous BS32 request nevertheless entered a device-side
spin during target replay. Resetting the 460-byte progressive epoch slice as a
captured layer-0 node, with an explicit main-to-alt-stream dependency before
draft publication, did not resolve the spin. The problem is consequently not
just eager prefill overwriting the initial epoch. Reusing `ca_comm.buffer`
aliases storage and synchronization state used by the existing per-layer
custom-all-reduce graph; the two protocols cannot safely coexist in the same
replay even when their first-use ordering is explicit.

The request produced no benchmark result and was terminated. The service-only
branch was removed again, leaving native AR and the accepted DSpark service
path unchanged. The accepted E2E checkpoint remains 1646.6 tok/s (best observed
1648.3 tok/s); the 120.344 us/layer standalone oracle is still promising but is
not a production result.

Do not retry either model-local graph allocations or `ca_comm.buffer` aliasing.
The next admissible integration requires a communicator-owned, separately
registered progressive arena whose capture and replay lifecycle is tested in a
minimal full-backend multi-stream harness before another complete model load.
