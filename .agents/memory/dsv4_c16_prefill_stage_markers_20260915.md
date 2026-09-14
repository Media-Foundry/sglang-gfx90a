# Original V4 TP8 C16 prefill: coherent stage envelopes (2026-09-15)

## Accepted performance is unchanged

`50891506eb` remains the production checkpoint: C16 native prefill ABBA
5298.152 -> 5569.784 input tok/s (+5.1269%), original weights, TP8/EP1,
1M logical KV pool. The new measurements below are diagnostic, not another
throughput improvement. See `dsv4_c16_prefill_empty_tiles_20260914.md` for
the accepted performance/correctness evidence.

## Failed profiler run, preserved rather than explained away

The separate `profile-B` process completed warmup and accepted `/start_profile`
with CPU/GPU activities. The measured request then stopped progressing after
two prefill forwards, alongside repeated ROCm queue-interposition asynchronous
signal waits. No completed trace was exported. GPU snapshots showed GFX busy,
UMC idle and about 17 GB free per GCD; these observations do not prove OOM,
hardware failure, or a particular model kernel fault.

Only the verified owned controller/service was stopped; the stop record has no
remaining children. A sibling gdb attach failed and noninteractive sudo required
a password. No global ptrace policy was changed. This run is excluded from
all timing results. It motivated in-process markers, not a claim that profiling
is generally impossible on this machine.

## Replacement diagnostic contract

`SGLANG_DSV4_DEBUG_PREFILL_MARKERS_DIR` is unset by default. With it unset,
the eager-runner decorator returns the original callable (no wrapper overhead).
When explicitly set, the collector is restricted to original V4, TP8/EP1/CP1,
native extend, M8192..65536. It reuses existing model/attention/MoE marker
slots. Decode/speculative execution does not emit records.

Each frame uses a 44x32 uint64 GPU matrix: 43 layer rows and one outer anchor
row. A same-stream asynchronous copy takes a private pinned snapshot; only a
background writer waits for completion. No scheduler-thread synchronize,
profiler interception, or new collective is added. Raw ticks and event times
are retained. This is still instrumentation, not a zero-overhead benchmark.

HIP reports `hipDeviceAttributeWallClockRate=25000` kHz on all eight ranks:
0.04 microseconds per `s_memrealtime` tick. This is not the GPU core clock.
Initial short synthetic calibration revealed FFI/event initialization overhead;
module/FFI priming and lazy-event materialization now precede the envelope.
The final implementation queries HIP's clock rate and uses events only as an
independent check, rather than deriving the clock rate from a short interval.

Validation:

- Three CPU scope/report tests pass (one unrelated pytest config warning).
- Single-GCD synthetic 43-layer lifecycle: two mutated inputs, exact outputs,
  full marker publication, and no non-extend record. This fake TP8 metadata
  smoke test is not evidence of TP8 performance.
- Real service: four C16 waves, one warmup plus three measured diagnostic waves;
  128 snapshots = 8 ranks x 16 forwards, all 43 coarse intervals valid.
- All 64 explicit input-ID echoes exact, zero prefix-cache hits, one completion
  token each. This short run does not establish whole-model determinism.
- Launcher default resolved the accepted prefill empty-tile flag to 1 without
  an explicit override; no fixed-projection or precision diagnostic was enabled.
- Warm GPU-event/realtime envelope ratios: 0.99999615..1.00001684.
- Service was stopped through owned-PID/birth validation; all eight GCDs free.

## Analysis method and results

For each of the 12 warm forwards, select the rank with the longest realtime
outer envelope, then keep **all stage intervals from that same rank**. Never
sum independent rank-wise stage maxima. Across ranks, a frame's longest-minus-
shortest envelope was at most 4.372 ms. Absolute clocks are not assumed to be
synchronized: this is a coherent envelope decomposition, not a reconstructed
global multi-stream critical path. Every selected frame closes exactly when
inter-layer gaps and outer work are included.

Mean over three waves (four forwards/wave):

| Span | seconds/wave | Meaning |
|---|---:|---|
| Attention-side MHC/Norm | 4.174 | boundary before attention; includes waits |
| Attention preparation | 5.229 | Q/KV, indexer and compressor chain |
| Main sparse attention | 2.749 | attention-call envelope |
| Attention output projection/collective | 1.601 | output chain, not pure AR |
| FFN-side MHC/Norm | 4.246 | boundary before MoE; includes waits |
| MoE/collective | 5.370 | complete MoE-call envelope |
| Whole GPU forward envelopes | 23.481 | includes remaining gaps and outer work |

Observed HTTP wave durations: 23.57284 / 23.57470 / 23.56872 s. These are
instrumented request timings, not an additional A/B throughput claim.

Nested subdivisions (already included above; **do not add again**):

- Routed stage: 4.370 s/wave, including current CK helper's full execution.
- MoE output collective span: 0.757 s/wave, including rank arrival waits.
- QKV projection: 0.550 s/wave.
- Q/KV preparation **plus indexer**: 4.053 s/wave (10->15).
- Core compressor: 0.625 s/wave (15->14).

Indexer alone is **not** isolated by the current slots: native prefill does
not populate decode-specific 11/12/13 preparation boundaries. Do not assign
all 4.053 s to logits or query projection.

The marker metadata `rows` is the eager runner's incoming row count, before
`load_batch`; scheduler logs may show rounded M32768. Individual real frames
can contain 32766/32767 rows. Retain both contracts rather than silently naming
all query tensors M32768.

## Consequences for the reviewer's proposal

The empty-tile recommendation has already earned the accepted +5.13% E2E
improvement. The new data also supports the warning against extrapolating old
MoE-dominated profiles: routed work here is roughly 19% of the envelope, while
the two MHC/Norm boundaries together are 8.420 s (about 36%). This does not yet
prove the MHC arithmetic itself consumes all of that span.

Before writing multi-query K reuse, add narrower default-off boundaries around
the actual prefill indexer query producer/logits/Top-K and the MHC sub-steps.
Audit actual large-M MHC dispatch as well as timing. Historical TP4 token-row
ownership and BF16-GEMM MHC had wins, but TP8 and current numerical contracts
are different. A TP4 hipBLASLt solution-5870 micro win was previously rejected
at E2E; do not repeat it as a new untested discovery. Do not globally enable
BF16 MHC or change precision to make this diagnostic look faster.

Multi-query K reuse and query-owner logical-index exchange remain bounded
candidates, not measured benefits. Preserve exact causal selection, tie order,
full compressor/cache updates, 1M capacity, and native decode isolation.

## Reproduction and evidence

Directory: `.agents/experiments/dsv4_tp8_c16_empty_tiles_20260914/`.
Run `capture_markers.py` only after checking GPU ownership and choosing a fresh
artifact directory; it deliberately refuses existing output. Analyze completed
data with `analyze_markers.py`; package with `package_markers.py`.

Archive: `marker-evidence.tar.gz`, 165 files, 2,837,300 bytes.
SHA256: `055bd5ebf4d34992bbbe2c4ad733b09a2337f768b4143255847dd6943ca55446`.
The manifest lists each member's size and SHA256. No model weights or tensor
dumps are included. Raw failed profiler artifacts remain separately labeled.
