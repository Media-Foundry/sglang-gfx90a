# TP8 AR four-block oracle and mutating-chain validation

2026-09-08, follows27e6b65b50. No production path or AIter library changes.
Owned service PID3611667 retained with1M KV. AMD PID ownership checked before
isolated eight-rank job; no service benchmark concurrent with the oracle.

Extended the existing shim's allowed geometry to4blocks. Compared shim16
against shim4, same legacy two-stage algorithm,512threads, BF16[32,4096],
same rank order and synchronization. Signal-size check against loaded AIter
passed; normal/integer rank-local mutations100/100 exact on all8ranks,
max_abs0, ten additional candidate replays per mutation stable.

Five graph ABBA cycles,200replays/sample,20warmups; slowest rank per sample:
-16blocks trimmed27.8111037612us, median27.8083503246us;
-4blocks trimmed25.2594521642us, median25.1593196392us;
-saving2.5516515970us, approximately9.17% latency reduction.

Additional graph:32 consecutive collectives with a distinct rank-local
`rank + step` input fill before each operation.1000 graph replays. Each step's
last-replay output compared to the exactly representable expected sum
`28 + 8*step`, all8ranks exact.8MiB transient output storage per rank, no
model/weight/KV cache. This stresses input overwrite/exit ordering, but only
checks outputs after the final replay, not every historical replay. Do not
overstate it as exhaustive race proof.

## Service contract audit before integration

`aiter.dist.device_communicators.custom_all_reduce.CustomAllreduce`:
-`custom_all_reduce` returns zeros_like during graph-context eager warmup;
-during actual capture it calls all_reduce(registered=True);
-outside capture, unregistered inputs are copied through self.buffer;
-capture context exit invokes register_graph_buffers.

Header get_buffer_RD records an unregistered graph input only while stream
capture is active, otherwise throws. The independent shim invokes this same
method via the existing public allreduce template. A prospective native-M32
adapter must preserve the parent custom_all_reduce wrapper, warmup allocation,
post-capture registration and unregistered eager copying. Do not replace all
all_reduce calls indiscriminately. Preload/validate the shim outside capture.

No service integration or E2E claim yet. Future selector must be default-off,
native DSV4 TP8/EP1 M32 only; leave C1/prefill/spec untouched and1M KV intact.
Raw rank-max samples, all-rank mutation/chain witnesses and log digest are in
the adjacent JSON. Command is prior oracle plus:
`--candidate-blocks 4 --shim-baseline --chain-length 32`.

Benchmark now aborts before timing if any rank's mutation/stability check
fails, rather than printing timing alongside a failed exactness result.
