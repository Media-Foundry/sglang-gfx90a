# H16 head ownership: real eight-rank component win; not deployed

Original V4-Flash TP8, original weights. Default accepted C16x8K remains
8384.698465 input tok/s. This experiment has NOT established a new HTTP result
or a whole-model drift fix. No production attention selector was enabled.

## Compute versus exchange

Existing stages1 two-bank sparse prefill uses an M16 head tile with eight real
heads on each TP8 rank. Pair two adjacent physical GCDs, assigning each half of
the query rows all16 heads, then restore original per-rank H8 ownership.
Each query keeps exactly its own index list/order/multiplicity, sink and causal
boundaries. This is not adjacent-query shared Top-K or changed TP weights.

Single physicalGCD4 compute-only screen (synthetic, independent nonzero heads):

| Family / M | H8 full-M ms | H16 half-M slowest half ms |
|---|---:|---:|
| C128 /8192 |1.271436|0.696533|
| C4 /8192 |3.777045|2.003745|
| C128 /32768 |5.374572|2.838307|
| C4 /32768 |16.444575|8.553770|

Conventional actualGCD4/5 Q/O RCCL sends, packing/cat/copies erased most gains:
C4/M32768 16.419292->15.154816ms; C128/M32768 5.364107->9.347052ms.
Do not omit these negative results when reporting the compute-only speedup.

## Direct peer loads/stores

Experimental Triton core retains the same QK/softmax/PV arithmetic, but reads
two H8 Q allocations and stores each H8 output to its owning allocation.
Timing includes full local Q staging plus two pair RCCL all-reduce fences.
Sink exchange is outside timing (static model parameters). No IPC setup is
included in steady timing. Setup/teardown still need production ownership.

PyTorch IPC metadata honors tensor and storage offsets. Test tensors deliberately
start64 elements into their storage; receiver opens IPC in its own device
context. This is NOT direct hipMalloc. There is no atomic queue/grid spinlock.
Static Q/O buffers at M32768 total about512MiB/GCD; not a service peak VRAM
measurement and not an accepted capacity cost against the1M pool.

Four pairs run concurrently on physical0/1,2/3,4/5,6/7. Each ABBA sample takes
the slowest of all8ranks, then median across threeABBA cycles; five calls/sample.
`peer-tp8-clean.json` synthetic results:

| Family / M | H8 ms | Peer chain ms | Speedup |
|---|---:|---:|---:|
| C128 /8192 |1.273732|1.525873|0.835x (loss)|
| C4 /8192 |3.776133|2.753158|1.372x|
| C128 /32768 |5.375696|4.359514|1.233x|
| C4 /32768 |16.430287|9.894531|1.661x|
| C4 /32767 |16.428304|9.931457|1.654x|

All8ranks pass five mutation cases byte-exact, including sentinel indices and
odd-M ownership. Banks use independently permuted physical slots per rank.

## Real model capture and replay

Default-off layer20 capture occurs AFTER the accepted live attention call.
Only live CSR spans are canonicalized; physical IDs are ordered by first
occurrence, preserving repeated occurrences and sentinels. The live model
reruns the canonical representation and requires byte equality before saving.

Successful fresh TP8 C16 capture PID2004545: actual M32767, lengths
[8192,8192,8191,8192], zero prefixes, original Q stride[4096,512,1]. Canonical
prefix bank8188 rows and extend bank32767 rows; all8ranks' banks, CSR arrays,
input IDs and positions have identical completeSHA256. Q hashes differ across
all8ranks as expected from head sharding.16 prompt echoes/cache checks and France
passed. Service stopped before GPU replay.

`peer-real.json`: complete file/tensor hashes checked before loading. Baseline
H8 replay equals captured production output. Five mutations pass on all8ranks,
then100 eager poisoned-output replays pass exactly. These are NOT HIP Graph
replays. Restore original real inputs before timing.

**Real layer20 M32767:16.5545868->9.9052532ms,1.671294x**, slowest8rank ABBA.
This single-layer oracle does not prove KV identity at every layer, mixed-prefix
safety, graph-safe lifetime, all43-layer performance, or full-model repeatability.

## Failures retained

- First peer pointer-selection version failed Triton make_ttgir type assertion
  before execution. Replaced scalar-pointer selection with explicit masked
  loads/stores. `failed_peer_kernel.py` preserves the failed source.
- Initial IPC teardown warned that producer exited before shared tensors were
  released. Releasing receiver references alone was insufficient. Clean version
  releases receiver mappings AND exported sender views/backing tensors with
  pair/global completion before ipc_collect/process-group destruction.
  `peer-tp8-clean.log` has no such warning. Earlier warning drivers/results are
  retained, not overwritten or reported as clean runs.
- First real capture failed in the PROBE: it inspected the full CSR allocation,
  including unused trailing IDs, then asserted bank bounds. Actual attention had
  completed. Retry uses indptr[0:rows+1] to slice/rebase the live span. Unit test
  specifically covers garbage tails/nonzero pointer starts; it does not change
  production attention selections. Both capture directories are retained.

Next gate: explicit runner-owned peer allocation/lifetime and all-layer KV
contract, default-off native large-prefill scope, then actual1M-pool service
correctness and ABBA. Never enable the losing C128/M8192 case blindly.

Artifacts: `.agents/experiments/dsv4_c16_h16_exchange_20260915/`.
Large tensor fixtures remain local; metadata records complete tensor/file hashes.
