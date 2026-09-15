# Paired-column FP32 pre-mix: guarded integration, service result pending

Independent-column candidate source and component evidence were committed at
4ee7ff12ab. M32768 pre-mix 7.63918 -> 4.98918 ms, approximately 34.7% less time,
not an E2E result. Per-row K1024 reduction and FP32 arithmetic are retained.

Production integration uses a new default-off MIX_PAIR_COLUMNS flag. It additionally
requires the existing mix reuse scope, original V4, TP8/EP1/no CP/PP/DCP, ordinary
EXTEND, non-draft/no speculative/TBO, no capture, group8 and M8192..65536.
The older reuse predicate is not tightened as part of this experiment. No new
tensor workspace. Existing batch-one FP16 Fn fused-tail priority remains unchanged.

Six CPU tests passed. Isolated physical GPU4 production-wrapper oracle completed
(integrated.json): M1/128/8191/8192/32767/32768/65536/65537, group4 and8,
exact outputs for flag off/on. Counted real kernel launches prove positive and
negative dispatch; large-M actual MHC caller also passed ten random x/Fn mutations
per shape, all FP32 output bits equal, scope contexts restored. These are not
fresh full-model activation checks or a claim that global service drift is fixed.

Fresh A1/B1/B2/A2 C16x8K service trial has started (sweep session35562), 3 formal
waves per leg and separate answer review. A1 PID1543098 passed readiness and
France; first formal wave6959.45 input tok/s. No candidate E2E result yet.
Keep default off until the complete result is accepted.

Extended CPU regression suite:45 passed, with one unrelated pytest asyncio_mode
warning. The initial run found the old AST-execution test fixture omitted the new
module-global pair predicate; adding its default None fixed the fixture without
altering production code. Three separate client tests pass, checking first-token
versus drain timing and rejecting wrong input echoes or empty output IDs.
