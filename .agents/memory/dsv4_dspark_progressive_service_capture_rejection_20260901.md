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
