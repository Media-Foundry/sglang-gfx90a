# Direct-activation wo_a: C32 portability rejected, 2026-09-08

User requested the previously unperformed C32 check for the ~0.2% C1 estimate.
That estimate was never an E2E measurement. No candidate-on C32 service result
existed. Current TP8 grouped selector admits G1 only at M1 and model call-site
also requires batch_size1/native decode. M32 falls back to einsum, so simply
enabling direct activation reads in the M1 kernel cannot speed up resident C32.
Batch drain to M1 is a separate possible tail effect, not C32 kernel acceleration.

Before widening the selector, ported the saved direct-X patch into an isolated
oracle header; production HIP header, model and Python selectors are unchanged.
No service restart, no graph recapture, no checkpoint modification. Existing
server2620634 on localhost30011 remains running with runtime-M repair.
Checked amd-smi process ownership; isolated tests used physical GPU4 with idle
service. Extra storage: 43 independent 8MiB BF16 matrices for cache-aware timing.

`scripts/rocm/bench_dsv4_tp8_woa_direct_m32.py` tests M1 and M32, G1,N1024,K4096,
same rows1/unroll2/waves4 and reduction order. For each M: 100 activation mutations,
weight mutation every25, 10 graph replays per mutation. Direct versus staged
GEMV exact, finite and replay-stable in 100/100 cases. Equality versus einsum
is a separate metric in JSON: GEMV and GEMM need not share a reduction tree.

Three ABBA cycles per staged/direct-versus-einsum pair, 43-node bursts x10;
median component timings in microseconds:

| M / memory regime | current einsum | staged GEMV | direct-X GEMV |
|---|---:|---:|---:|
|1 / repeated weight|30.380|7.072|6.685|
|1 / 43 weights|32.645|9.725|9.102|
|32 / repeated weight|32.001|88.073|122.423|
|32 / 43 weights|33.197|90.472|124.275|

M1 improvement relative to staged is reproduced (~0.623us/layer, only ~0.22%
of current C1 wall time if fully critical). M32 direct is ~3.74x the current
einsum time and is also slower than staged GEMV. Independent token GEMVs repeat
the weight scan and do not gain GEMM's multi-row weight reuse. Removing staging
does not solve that decomposition mismatch.

Decision: reject expansion to C32; do not perform an expensive service ABBA
whose unchanged M32 selector cannot exercise the candidate. No candidate C32
E2E number is claimed. Current ~922 complete-request / ~963 resident-decode
tok/s belongs to the prior runtime-M repair, not direct-X. A potential C1-only
service experiment remains unperformed; ~0.2% is still a theoretical estimate.

Reproduce with DS Python, HIP_VISIBLE_DEVICES=4 and normal repo PYTHONPATH,
script above, --output /tmp/dsv4_tp8_woa_direct_m32_20260908.json.
Adjacent JSON preserves samples and correctness counts. Production guard was
also exercised with mocked JIT: M32 returns None before loading a module even
when allow_single_group=True. No production defaults changed.
