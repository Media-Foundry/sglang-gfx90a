# TP8 BS1 attention stream screen and nonblocking diagnostics

## Scope

Continue after `6b7f55dece`. Original weights, TP8/EP1/no-A2A, native AR.
Keep M36864 BF16-CK prefill, C4 trivial skip, 131072-token pool, 0.80 memory
fraction and graph tiers 1/2/4/8/16/24/32 unchanged. Scheduler overlap and SBO
remain enabled. Only loopback port 30011 is used.

## Diagnostic isolation

1. Marker writes only (`TRACE_LAYER=20`, `TRACE_LOG_EVERY=0`) completed France
   and six 256-token code generations, with the same printed hash prefixes as
   baseline. A later teacher-forced eager prefill stalled around an M142 GEMM
   first-use log. GPU5 was 100% active while the other seven were idle. This
   does not prove CPU readback was the sole cause of the previous stall.
2. Replace blocking `.cpu().tolist()` inside graph execution with a single
   pinned snapshot plus a CUDA/HIP event, polling only with `event.query()`.
   Pending snapshots are neither read nor overwritten. Replays are tagged so
   samples can be matched across ranks despite a skipped poll.
3. `SGLANG_DSV4_GFX90A_REALTIME_TRACE_GRAPH_ONLY=1` resolves the JIT module during
   warmup but emits timestamp kernels only during capture, excluding eager
   prefill from instrumentation. Default remains off.
4. Async graph-only logging collected **96 complete eight-rank samples**.
   All six 256-token generations completed with baseline hash prefixes.
   However, a later teacher-forced prefill again stalled (M46 first-use log).
   Therefore diagnostic instrumentation is **not fully validated across
   prefill/decode transitions**. Neither this nor the earlier stall establishes
   a numerical kernel failure or a hardware fault.

Raw diagnostic logs:

- `/tmp/dsv4_tp8_bs1_marker_writeonly_20260907.log`
- `/tmp/dsv4_tp8_bs1_marker_async_20260907.log`

The original harness wrote only at the end, so those interrupted diagnostics
did not preserve complete output-ID artifacts. Do not upgrade matching printed
hash prefixes to a full teacher-forced parity claim. The harness now atomically
checkpoints completed evidence outside request timing, with `status=running`
until all checks finish.

## Layer20 C4 timing evidence

See `dsv4_tp8_bs1_marker_async_20260907.json`. Historical 0.04 us/tick calibration;
nonzero/monotonic slots required. Medians of per-replay eight-rank maxima:

| Span | Instrumented us |
| --- | ---: |
| Attention MHC/norm | 27.36 |
| Attention prepare | 101.60 |
| Sparse attention core | 28.16 |
| Output projection + collective | 64.32 |
| FFN MHC/norm | 24.48 |
| MoE + collective | 103.68 |
| Whole layer | 338.88 |

Do not sum component rank-max medians as a critical path. The single C4 layer is
not the average of all 43 layers. Markers themselves also add overhead.

Source inspection confirms attention preparation is **serial** at TP8 BS1;
ROCm multistream in the profile enables MoE/SBO, not this attention predicate.
Serial prepare intervals included q-a projection ~15.20 us, q norm ~6.72 us,
q-b ~10.72 us, fused QK/store ~7.20 us, indexer/compressor ~30.56 us and core
compressor ~29.44 us. These are instrumented stage spans, not pure GEMM times.

The active MoE path is `forward_normal_dual_stream`, so slots 16–17 mean router
(~7.68 us), 17–18 TopK (~14.56 us), 18–19 routed experts (~52.08 us), and 23–24
collective/arrival (~22.72 us). Do not label these using the serial MLP mapping.

## Narrow native-AR candidate and negative results

A temporary, default-off `SGLANG_DSV4_GFX90A_TP8_BS1_ATTN_MULTISTREAM` gate
allocated the existing HIP preparation streams for TP8, but selected them only
for unified-KV, native DECODE, one row, one request, and graph capture.
A CPU test evaluated the actual source predicate across TP4/TP8, BS1/2/32,
decode/extend/verify/draft/idle and compressed/noncompressed cases. Other graph
tiers and prefill could not leak through the generic multistream predicate.

| Screen | Median output tok/s | Validation |
| --- | ---: | --- |
| Prior uninstrumented control | 76.430 | full IDs and teacher probes exact |
| C4 + C128 overlap | 69.204 | France; 9/9 full 256-token IDs; 6/6 teacher probes exact |
| C4-only overlap | 67.804 | France; 9/9 full 256-token IDs; 6/6 teacher probes exact |
| Final unchanged control | 76.109 | France; 9/9 full 256-token IDs; 6/6 teacher probes exact |

Each candidate had a fresh service, three real code tasks, one warmup/task and
three measured rounds/task. Teacher checks compare identical supplied
continuation IDs and returned input logprobs/output top-logprobs against the
prior fixed baseline, not independently generated prefixes.

These are negative screens, **not a formal same-candidate ABBA speedup claim**.
The C4-only candidate is distinct from the first candidate, so they must not be
called the two B arms of an ABBA. The final unchanged control restored speed:
C4+C128 is 9.07% slower, C4-only 10.91% slower than that closing control.
The closing control retains one measured 57.21 tok/s outlier (diverse-03, rep1)
and a 58.29 tok/s warmup. A similar roughly 57–58 tok/s isolated sample appeared
in previous baseline runs. Its cause is not established; do not discard it or
claim perfectly stable per-request latency from the median alone.

`dsv4_tp8_bs1_attention_stream_screen_20260907.json` includes all measured
rates, completion hashes, full representative code outputs/IDs, artifact
digests, and explicit per-probe comparison results for both candidates and the
closing control. Raw fixed-prefix logprobs remain in the referenced `/tmp`
artifacts; the baseline reference is the same as the preceding migration.

Conclusion: do not enable this schedule. Stream fork/join and resource
contention are plausible explanations but were not independently measured.
This extends the old TP8/DP2 negative result to current TP8/DP1 shapes. The
temporary selector and its test were removed; user source changes were
preserved. No weight, precision, attention selection, or prefill path changed.

Candidate artifacts:

- `/tmp/dsv4_tp8_bs1_attn_multistream_B1_20260907.json`
- `/tmp/dsv4_tp8_bs1_attn_multistream_c4_20260907.json`
- same stems with `.log` for fresh service logs.
- Closing control: `/tmp/dsv4_tp8_bs1_control_A2_20260907.{json,log}`.

The normal, uninstrumented service remains resident at `127.0.0.1:30011`.
Large-prefill settings were retained; P32/D32 were not remeasured this turn.

## Next direction

The new evidence argues against adding more attention side streams. Examine
the ~64 us output projection/collective boundary and the exact router/TopK
producer path before revisiting low-level FP4 decode, while keeping this serial
attention baseline. Diagnostic eager-prefill stalls remain a separate issue;
no production default is changed based on the diagnostic traces.
