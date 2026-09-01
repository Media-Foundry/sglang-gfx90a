# DSpark progressive M128 secondary custom-AR integration audit (2026-09-01)

## Motivation

The exact standalone progressive M128 oracle saves 120.344 us/layer, but all
production attempts that shared the primary SGLang custom-all-reduce instance
failed: external graph allocations crashed at `hipStreamEndCapture`, while
aliasing the primary `ca_comm.buffer` captured successfully but spun on the
first real target replay. The primary communicator's payload, metadata,
registration table and synchronization epochs must not be reused.

## Feasible isolation unit

A second AIter `CustomAllreduce` instance is the smallest credible isolation
boundary. Every instance owns an independent gfx90a uncached metadata
allocation, eager payload buffer, rank-data table, C++ object, registered
buffer map, graph-unregistered-address list and IPC-handle map. Construction is
not process-global, and `register_graph_buffers()` operates on the instance's
own pointer. The existing TP CPU/Gloo group may be reused, provided every rank
constructs and registers the instances in the same order.

For M128 BF16, instantiate the secondary communicator before any graph capture
with `max_size=1<<20`. This adds roughly 10 MiB/GCD rather than the default
large allocation. The progressive signal workspace must be a separate
`allocate_meta_buffer(sync_bytes)` uncached allocation registered only with the
secondary communicator. Do not use either communicator's ordinary payload
buffer as MI200 polling metadata.

## Capture lifecycle

The secondary `capture()` context must wrap the complete decode graph capture
session, alongside the primary capture lifecycle, and exit only after all
shape captures have left device stream capture. Its exit performs CPU
broadcast and IPC graph-buffer registration, which is unsafe from inside
`torch.cuda.graph(...)`.

Full-backend capture performs eager warmups while the communicator reports
`_IS_CAPTURING=True` but the stream is not yet capturing. The progressive
wrapper therefore needs a fail-safe warmup branch that returns an
allocation-compatible placeholder whenever:

```python
secondary._IS_CAPTURING and not torch.cuda.is_current_stream_capturing()
```

Only actual graph capture may invoke the progressive peer-read kernels.
Teardown must close the secondary communicator before destroying its CPU
process group.

## Required harness before another model load

Build a minimal full-backend-style four-rank harness using physical GCDs 4--7:

1. primary communicator performs the unchanged production M128 all-reduce;
2. secondary communicator serially performs shadow progressive M128 work on
   independent payload and uncached sync storage;
3. both communicators exit their capture contexts and independently register
   nonzero graph address sets;
4. mutate input for 100 replays, require primary and shadow outputs bitwise
   equal on every rank;
5. require 1000 graph replays without stale data, spin or deadlock.

The first harness must launch the two collectives serially. Separate registry
state prevents protocol aliasing but cannot prevent two simultaneous polling
kernels from exhausting CU progress. Cross-stream overlap is admissible only
after the serial harness passes and must preserve CU capacity for both
protocols.

No production model selector should be restored before this harness passes.
Native AR remains untouched.
