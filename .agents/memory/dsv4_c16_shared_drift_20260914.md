# C16 shared-expert drift: frozen-input causal decomposition

Base `a1bae27809` on fork main. Original V4-Flash, TP8/EP1, native AR,
original checkpoint and 1M logical KV. Existing C16 throughput defaults stay
unchanged. This is correctness investigation, not a new throughput result.

## Complete shared chain reproduced

`shared_oracle.py` restores real layer0 normalized FFN inputs from the
completed `layer0-stable-qkv-wqb-wob` trace at their actual M32768/M32767 offsets.
The 153 common case15 rows are identical before the shared expert. It reads
actual FP8/E8M0 checkpoint w1/w3/w2, reproduces block128 BF16 caching, and uses
the TP8 shards: merged gate/up[512,4096], down[4096,256]. Untouched rows are
zeros; a service-output reproduction assertion protects the interpretation.
All eight shard fixtures execute on physical GCD4, not eight GPU processes.

The ordinary F.linear -> eager BF16 bounded SiLU/multiply -> F.linear chain
reproduces **all retained service values on both arms for all8 shards**.
The activation specifically keeps production's BF16 clamp, SiLU and multiply
roundings; it is not an FP32 fused mathematical substitute.

| Variant | Gate/up shifted elements | Intermediate shifted elements | Shared output shifted elements | Full-chain component median |
| --- | ---: | ---: | ---: | ---: |
| library both | 9 | 9 | 571 | 2.3525 ms |
| fixed gate only | 0 | 0 | 42 | 2.5019 ms |
| fixed down only | 9 | 9 | 531 | 2.2943 ms |
| fixed both | 0 | 0 | 0 | 2.4750 ms |

Fixed variants use M128/N128/K128 with8 waves. These are isolated component
medians, not service ABBA; no speed claim is inferred from the small down-only
timing difference. Both GEMMs need fixed ordering on this fixture. Changing
only the gate still leaves down-GEMM row sensitivity. Fixing only down cannot
repair already-different gate inputs. The fixed-both chain is about5.2% slower.

The integrated shared entry matches the offline full output and passes100
varied-row perturbations for each of eight shards (800 total). Nine CPU
contract tests pass. Fixed ordering is not old-GEMM bitwise parity.

## Narrow integration and pending service validation

`SGLANG_DSV4_DEBUG_PREFILL_SHARED_STABLE` defaults OFF. Only ordinary
`DeepseekV2MoE.forward_normal` supplies forward_batch to the shared wrapper's
new optional argument. The existing native/gfx90a/TP8/EP1/M8192..36864 guard
must pass, and both shared projections must be unbiased cached BF16 TP8 with
no down all-reduce. Unexpected shapes, pre-quantized input or limit fail
explicitly. Other callsites, small-M, decode and draft do not opt in.
The numerical candidate preserves every selected routed expert and weight.

The performance harness explicitly zeros this flag. The diagnostic service
adds only shared stabilization to the prior QKV/wq_b/wo_a/wo_b + FP32 AR
profile; router remains unchanged to keep the service comparison single-variable.
Completion and measured first-divergence findings will be appended after the
owned warmup/A1/B1/A2 job finishes. Sampled layer0 closure, if achieved, must
not be advertised as all-layer or whole-model determinism.

## First service attempt did NOT validate the repair

The `layer0-stable-qkv-wqb-wob-shared` job completed warmup/A1/B1/A2 and all64
input echoes. However, the service logged **zero stable-shared kernel hits**,
and the old eight FFN output row differences remained unchanged. The strict
`validate_shared.py` assertion failed at ffn_shared. This is a wiring/scope
failure, not evidence against the component kernel, and not an accepted repair.
No production default has changed. Preserve these logs and negative results.

Added a runtime environment/source-hash witness and a warmup hit assertion:
future shared trials stop immediately if the selector was never executed.
An opt-in once-per-layer scope log prints actual M, TP, model flag and whether
forward_batch reaches the shared wrapper. `layer0-shared-scope-probe` is the
new owned investigation; do not count a pending probe as a successful result.

Meanwhile the completed component `--axes` test isolates M from row placement:
at constant M, the same9 gate /9 intermediate /571 shared-output changes
remain. At fixed rows, changing M alone produces zero retained differences.
Fixed gate only leaves42 down differences; fixed down only leaves531 upstream
propagated differences. Fixed both has zero differences on both axis checks.

The scope probe found the concrete omission: runtime shared calls had
M32767/TP8/V4=true but **forward_batch=None**. The actual eager prefill shared
work is launched from the SBO `_pre_combine_hook` inside `forward_normal`,
not the two ordinary shared callsites initially patched. That third call
did not forward the batch. The probe stopped after warmup via the new hit
assertion; this is an intentional failed check, not a GPU fault.

The hook now forwards the owning forward_batch, preserving its existing
alt-stream wait and post-combine join. The CPU wiring test now checks all
three callsites including the nested hook. `layer0-shared-wired` reruns the
same test with this forwarding correction; no other numerical change was
made. The two non-hitting attempts remain in the evidence bundle.

## Correctly wired service result

`layer0-shared-wired` completed warmup/A1/B1/A2 and stopped its owned service
(PID1059770). All64 full input-ID echoes matched; completion IDs decoded to
the returned text. The requested flag and source hashes were recorded from
the live process environment. All eight ranks logged stable shared execution.

On every retained common-query sample, all attention, FFN-entry MHC/norm,
shared, routed and reduced FFN output values now agree across the placement
change. **FFN output changed rows:8 -> 0.** Full retained QKV/KV comparisons
also match. Same-order sampled stages are exact across all ranks.
`validate_shared.py` passes and enforces actual kernel hits before numerical
checks, so a no-op selector cannot pass this acceptance again.

One router-logit row still differs on each rank, without changing this hash
layer's sampled TopK IDs/weights or FFN outputs. Router invariance has not been
fixed. Complete128-token output matches are14/16 for both same-order A1/A2
and reordered A1/B1. Therefore layer0 sample closure does **not** imply full
model/sequence determinism. Earlier prefill groups, unobserved positions and
later layers remain outside this snapshot proof. Do not attribute those
remaining final-token differences to router alone without further evidence.

No new E2E throughput measurement is claimed from these tensor-dumping runs.
Production retains its previous numerical path; all stability selectors are
default-off. The accepted C16 throughput figures (including the 64K opt-in
+1.82%/TTFT trade-off) remain separate. Next work is to extend first-divergence
coverage past layer0 and seek lower-cost row-invariant projections before
considering any default change.

Evidence: `shared-evidence.tar.gz`,84 files,9,241,601 bytes, SHA256
`9e27e9286db62bb558d259277a471a7470b6a16096b44be10ff4e9ce849b8acd`.
Contains component/axis tests, both non-hitting service attempts and the wired
success, runtime witnesses, responses, logs and hashed activation manifest.
Full activation/weight tensors are not uploaded. Nine CPU contract tests pass.
