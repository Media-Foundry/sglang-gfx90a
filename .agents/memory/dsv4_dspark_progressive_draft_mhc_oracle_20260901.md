# DSpark progressive M128 draft-MHC oracle (2026-09-01)

## Result

A dependency-correct four-rank oracle now overlaps the next layer's draft-row
M96 entry MHC with the current layer's routed-M32 tail while preserving one
logical M128 TP4 collective epoch.

Formal physical-GCD 4--7 ABBA rank-max result:

```text
baseline:    618.583 us/layer
progressive: 498.239 us/layer
saving:      120.344 us/layer
gain:         19.455%
```

The 43-layer target pass budget is about 5.17 ms. This clears the predeclared
100-us/layer continuation gate and is large enough to justify production
integration.

## Dataflow

```text
shared M128 ready
  -> one progressive TP4 collective epoch begins
  -> draft M96 rows become ready
     -> next-layer M96 MHC post/pre + RMSNorm on consumer stream

routed M32 finishes on main stream
  -> main waits for the draft reduction stream
  -> fused shared+routed anchor publication
  -> anchor M32 reduction and the same collective epoch exits
```

The important replay fix is that the late anchor publication/reduction runs on
the main producer stream after `main.wait_stream(comm_stream)`. The previous
`main -> comm` captured HIP event produced stale anchor rows under changing
inputs even though fixed-input timing looked exact. A device-side spinning wait
was also rejected because it deadlocked graph capture/replay scheduling.

## Correctness and stability

- Original arithmetic and row order `[anchor,d0,d1,d2]` retained.
- Full M128 output, extracted M96 draft rows, and all MHC outputs were
  bitwise identical on all four ranks.
- 100 changing shared-expert input mutations: 0 failures on every rank.
- 1000 candidate HIP graph replays: bitwise stable on every rank.
- Physical GCDs 4--7 were idle according to `amd-smi` before the formal run.

The MHC output allocation required relaxed HIP capture mode in this standalone
multi-stream oracle. Production integration must use graph-owned/preallocated
state and must not rely on ad-hoc allocations inside capture.

## Next gate

Integrate only behind an explicit strict guard:

```text
gfx90a + DSpark + TARGET_VERIFY + TP4/EP1 + BS32 + gamma3/M128
```

Native AR and all other graph tiers must remain unreachable. Before retention:

1. capture the real service graph without relaxed allocator behavior;
2. pass BS1 France at least 3/3;
3. complete the frozen heterogeneous BS32 workload at 1024 tokens/request;
4. achieve at least 1729 tok/s E2E; otherwise remove the production selector.
