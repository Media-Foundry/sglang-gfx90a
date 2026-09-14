# First component attempt: harness failure, not a large-M result

Source revision: `efcf1eb71d` (oracle unchanged through `7229ac94b1`).
Command used physical GPU4, sizes1/8192, three mutations and six replays.
Run session57456 exited1. `screen-small.json` is the preserved partial file;
its `status:running` must not be interpreted as an active process or a
completed screen.

M1 completed: A/B/R outputs byte-exact, three mutation checks passed,
A and B each6/6 eager replays exact, row-permutation check passed.
The same-path allocating Python-boundary timings were0.210991/0.212223ms,
not an AR service speed claim.

The next shape M8192 failed during warmup, before any result was accepted:

```
mhc.py:3100: get_tp_group(), disabled=not is_allocation_symmetric()
parallel_state.py:1953:
AssertionError: tensor model parallel group is not initialized
```

The FP32 path evaluates the group accessor even though symmetric allocation
is disabled in this single-GCD test. Existing standalone MHC oracles use a
None group and disabled symmetric allocation for this no-collective boundary.
The retry follows that pattern in every arm and restores the accessors in
`finally`. No production math, dispatcher or service configuration is changed.
No M8192 speed, correctness or replay result follows from this failed attempt.

## Second attempt: unmatched Sinkhorn reference

Session60043 exited1 after M1 again passed. The M8192 warmup ran, but the
initial B/R exact-output assertion failed before timing/mutation acceptance.
`screen-small-v2.json` retains only M1 and is not a complete screen.

Source inspection found another batch-size-dependent math choice:
`hc_split_sinkhorn` uses the environment Sinkhorn iteration override8 for
batch1, but20 for batch2. A and B have batch1; the original R had batch2.
Thus the original reference was not mathematically matched. No output deltas
were saved before this assertion, so the attempt does not quantify the error
or establish that this is its sole source. The next attempt keeps R20 as a
diagnostic and adds R8 with only the Sinkhorn hint aligned; B/R8 must still
pass byte equality. This does not weaken the acceptance into a tolerance test.

Pre-fix v2 source and partial JSONs are in `failed-v2-evidence.tar.gz`, SHA256
`71db3a3cc1c98751b35a3886c8188bb6773ba2ca3a05f27ab26165220a7a5522`.
