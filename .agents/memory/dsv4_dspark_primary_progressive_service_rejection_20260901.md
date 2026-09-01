# DSpark primary-progressive M128 service rejection (2026-09-01)

## Accepted state remains unchanged

- Physical GCDs: 4--7
- TP4 / EP1 / no A2A, original weights, gamma-three DSpark
- Real heterogeneous BS32 requests, `stream_interval=1`
- Last accepted long-window E2E: about 1646.6 tok/s stable, 1648.3 tok/s best

No progressive result below is an accepted performance checkpoint.

## Two-stage AIter lifetime bug found

The saved `module_custom_all_reduce.so.pre-2stage-final-barrier` passed the BS1
France exact oracle but failed when the same request occupied the heterogeneous
BS32/M128 target graph. The fixed M128 payload uses
`cross_device_reduce_2stage`: it synchronized after reduce-scatter, but returned
immediately after all-gather while peers could still be reading its temporary
shard. A fast rank could therefore reuse the shard in the next layer.

Adding `end_sync<ngpus, true>` after stage-two all-gather restored:

- 43-layer primary-communicator oracle: 100/100 rank-distinct mutations exact;
- 1000/1000 graph replays exact;
- BS1 France: exact first nine tokens;
- real heterogeneous BS32 France: semantic Paris.

The launcher patch now carries this final lifetime barrier as part of the
gfx90a custom-AR correctness patch.

## Primary-progressive protocol result

A fixed TP4/M128 primary-communicator primitive was implemented in the external
AIter worktree. It reduces request-major draft rows M96 on the auxiliary
stream, publishes late routed M32 anchors, then closes the same primary Signal
epoch. Its standalone 43-layer graph oracle passed with max absolute error 0.

The same primitive was then guarded to DSpark TARGET_VERIFY, BS32, M128, TP4,
EP1 and enabled for layer 0 only. The full service captured and ran, but the
BS32 France sentinel consistently became:

```text
[671, 6102, 294, 8760, 14, 1008, 4987, 16, 270]
```

The following hypotheses were rejected; every candidate retained the same
failure:

1. missing routed scaling factor;
2. premature reuse of compact `anchor_output` (explicit clone tested);
3. duplicate graph IPC registration (pre-registered AIter buffer tested);
4. BF16 double rounding in the anchor join (the stock `add_(alpha=...)`
   operation was executed on the fixed buffer before publication).

A decisive dual-path graph ran progressive first and then the ordinary stock
M128 all-reduce, returning the stock result. France recovered semantically.
Therefore the progressive epoch does not corrupt the following primary
collective; the unresolved mismatch is confined to the progressive output
under the real SGLang graph. The production selector and environment switches
were removed. The standalone harness remains useful, but this route must not
be enabled or used for throughput claims until a real-layer tensor comparison
locates the first differing anchor/draft pack.

Artifacts:

- `/tmp/dsv4_primary_progressive_finalbarrier_oracle.json`
- `/tmp/dsv4_finalbarrier_control_bs1.json`
- `/tmp/dsv4_finalbarrier_control_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_scaled_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_stable_anchor_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_registered_buffer_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_stock_return_bs32_r1.json`
- `/tmp/dsv4_progressive_l1_exact_prejoin_bs32_r1.json`
