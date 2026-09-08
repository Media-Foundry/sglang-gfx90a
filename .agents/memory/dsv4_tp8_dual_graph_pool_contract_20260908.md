# Same-process TP8 graph crossover: collective/pool prerequisite

Baseline e7b461bffe, native service PID3468727 stayed resident with1M pool.
No model, scheduler, backend or service selector changes in this experiment.
Standalone torchrun uses all8 GCDs, gloo control plane, installed AIter
CustomAllreduce, with original service idle. AMD-SMI ownership checked before
each run; no foreign GPU process. No frequency, NUMA or dependency changes.

## Inspected production contract

GroupCoordinator.graph_capture wraps the entire shape loop in ca_comm.capture.
Installed AIter class is under /home/pc/pytorch/third_party/aiter, not a separate
/home/pc/Code/aiter checkout. Its capture context collects graph addresses and
registers IPC handles/offsets once at outer exit. FullCudaGraphBackend owns
graph/output dictionaries and its capture pool; each replay returns the output
paired with the selected graph. Extra candidate capture must occur inside the
same outer registration context, preserve attention post-warmup hooks, and
restore the original pool afterward. This audit alone is not a model test.

## Eight-rank oracle

New scripts/rocm/check_dsv4_tp8_dual_graph_pools.py captures:
- M32 A in baseline pool;
- C1 in the same baseline pool;
- M32 B in a separate pool.

Each forward creates an internal graph-pool producer tensor then AIter AR.
M32 uses the existing legacy AR flag, C1 the normal new path. Both M32 graphs
share external input buffers, matching prospective service usage. All graph
addresses register together at outer capture exit. Test alternates1000 replays
in A/B/B/A/C1/B/C1/A order, changing nonconstant per-column/rank inputs. Dyadic
inputs have an exact known BF16 sum, and output raw int16 bits are compared.
Graph identities and output pointers remain fixed.

First draft failed at iteration0 checking an *old* C1 output after replaying
M32 A. This was an overstrict test contract, not evidence of broken IPC:
the original two shapes share a pool and old output storage may be reused by
another graph's intermediates after its consumer completes. Corrected oracle
checks current output against the mathematical oracle and preserved snapshots
only across distinct pools. It separately counts same-pool old-output changes.
All8 ranks observed375 such same-pool changes over1000 replays. Never require
old shared-pool output values to survive a subsequent different-shape replay.

Both corrected runs exited0,1000/1000 replay iterations/rank exact, independent
pool outputs preserved, C1 graph identity stable. The initial run exited1;
its worker tree was gone before retry. No stuck GPU process was killed.

## Memory-accounted repeat

Per rank, after three captures and IPC registration relative to before capture:
- PyTorch live allocated delta535,552 bytes;
- PyTorch reserved delta4,194,304 bytes (4MiB);
- device free-memory drop44,040,192 bytes (42MiB).

The difference includes non-PyTorch runtime/capture/registration storage, not
just tensor storage; no counter decomposition attributes the whole42MiB to one
subsystem. Communicator initialization precedes the measurement baseline.
Do not report535KiB or4MiB as total graph overhead, or extrapolate this small
collective oracle to the full43-layer model. All transient test resources are
released at process exit. Raw per-rank evidence is in adjacent JSON.

```sh
HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang OMP_NUM_THREADS=1 \
 /home/pc/anaconda3/envs/DS/bin/torchrun --standalone --nproc-per-node=8 \
 --max-restarts=0 scripts/rocm/check_dsv4_tp8_dual_graph_pools.py
```

Next implementation gate: narrow default-off full-backend M32 dual capture,
separate pool/output ownership, idle-only rank-consistent switching, retain C1
graph and1M pool, account full device-memory delta, then complete real-request
correctness and same-process ABBA. No E2E gain or production readiness is claimed
from this prerequisite. Do not enable graph pool borrowing for the experiment.
