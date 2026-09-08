# Default-off native TP8 paired M32 graph wiring

Continues `91e54f29b3`. Not an accepted optimization or new speed measurement.

## Implementation

- `down_uniform_capture(bool)` uses a ContextVar, restores nested overrides on
  exceptions, and still requires the existing native M32, TP8/EP1 and exact
  geometry predicates. Outside capture the existing environment flag applies.
- Full backend calls the unchanged capture implementation through a wrapper.
  Only the exact decode runner opts in with
  `SGLANG_DSV4_GFX90A_TP8_M32_DOWN_PAIRED_GRAPHS=1`.
- Baseline graphs use the original pool. M32 additionally captures a candidate
  in a distinct pool through the tested transaction. Graph/output pairs remain
  associated; C1 uses its original baseline graph and output.
- Intended topology is native DSV4 gfx90a TP8/EP1, DP1/PP1, no DP attention,
  no speculative/draft, no memory saver/pdmux/compile/ragged/profile capture,
  with C1/M32 tiers and max-total-tokens=1048576. The standalone DOWN_UNIFORM
  flag must be off so eager execution remains baseline.
- `set_internal_state` accepts the sole boolean `dsv4_down_uniform_arm` key.
  Each worker uses all-rank idle/readiness consensus, device synchronization,
  a memory check and TP barrier before switching. Invalid or mixed requests
  are rejected. The default service has no pair and rejects this command.
- Extra full-device memory is capped at 1 GiB/GCD, with >=2 GiB free reserve.
  Admission requires >=3 GiB free. Capture checks the immediate delta; switching
  checks again after outer IPC registration, conservatively including subsequent
  graph tiers and other runtime allocations. No KV pool reduction is performed.

## Verification performed

- Actual helper/full backend imports passed in the DS environment.
- AST checks passed for changed implementation files.
- Existing down-uniform scope test now covers both environment states and both
  capture arms, M1/2/4/8/16/24/32/64, native/spec, extend/decode, nested failure
  cleanup, boolean-only arms and independent legacy AR flags.
- Seven mocked pair tests cover graph selection/C1 identity, remote-busy
  rejection, pre/post-capture memory rejection, late registration memory
  rejection, invalid arms/not-ready and cleanup.
- Seven capture-transaction tests remain passing.

## Still required

No service restart or GPU test in this step. Constructor validation, full-model
capture, cross-rank control and memory accounting need real startup validation;
CPU mocks do not prove these. Next: audit GPU PIDs, launch an owned paired
diagnostic service preserving the 1M pool, validate C1/C32 output correctness in
both arms, then fixed-corpus same-process ABBA. Keep this flag default-off even
if the kernel eventually wins: duplicate graphs are diagnostic overhead, not a
production speed feature. Do not claim C1 non-regression yet.
