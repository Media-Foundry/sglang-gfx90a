# Native TP8 down-consumer: C32 first, C1 conditional follow-up

The user clarified that M128 was not intrinsically speculative, then requested
a small C32 native-AR trial followed by C1 only if beneficial. We test actual
M32, not C32 padded to128, and never enable DSpark. V4.1 remains frozen.

Source base:57396b1389, with the narrow default-off native-M32 selector added
in this experiment. Original V4 weights, TP8/EP1/no-A2A,1M allocated logical KV
pool, graph tiers1/2/4/8/16/32/64, original best-AR launcher settings including
C1 wo_a GEMV. All GPU ownership checks use AMD-SMI; no unrelated owners found.

## Component result (not an E2E speed estimate)

On physical GPU4, current uniform-metadata down vs CTA16 consumer, synthetic
diverse/skewed routing and nonconstant physical E8M0 scales:

| Quant + down + fixed reduction | A, us | B, us | Time reduction |
|---|---:|---:|---:|
| Diverse |130.542|115.907|11.21%|
| Skewed |81.615|70.791|13.26%|

Each regime:100 input/routing/weight/scale mutations,1000 graph replays with
FP32 partial and BF16 output checked every replay. `torch.equal` passes;
this is numeric element equality, not a signed-zero-bit or real-layer oracle.
Do not count these as complete routed-stage measurements: gate/sorter omitted.

## Completed C32 service ABBA

Fixed32 real public-source requests,512 input tokens, natural EOS, maximum2048
output tokens. Each leg has two waves, each with a10--12s common resident
window. Prefill and drain are excluded from resident rate, retained in HTTP.
Separate A1/B/A2 processes; B1/B2 are legs within the B process. Actual M32
graph execution is logged on all ranks. B logs43x8=344 selected consumer
instances. Existing generic flags are forced0; only the new scoped flag changes.

| Leg | Resident tok/s | Whole HTTP tok/s |
|---|---:|---:|
| A1 |1047.763|714.964|
| B1 |1060.206|712.228|
| B2 |1061.203|696.613|
| A2 |1041.459|687.516|

Mean of leg medians: **1044.611 ->1060.705 resident tok/s, +1.541%**.
Return control changes-0.602%; B-leg difference+0.094%. Every B wave exceeds
every A wave, but this remains a small screen, not the earlier formal matrix.
Whole HTTP varies with output length/drain and does NOT establish the same
1.54% improvement; do not rename resident throughput as complete-request speed.

All256 measured output-ID sequences decode exactly to saved response text.
All three service France sentinels answer Paris. No response exceeds0.5 on
the explicitly defined last256-token duplicate8-gram-ratio screen. Readable
excerpts have no obvious collapse, but include unsupported code assertions;
they are not a factual correctness oracle. Full output repeatability remains
poor in BOTH arms (within-leg exact counts A1=0/32,B1=0/32,B2=1/32,A2=0/32).
Do not attribute this pre-existing batch/admission-sensitive drift to the
consumer, or claim whole-model bitwise equivalence from the component test.

## C1 scope

The existing M1 direct down kernel already quantizes intermediate BF16 into
CTA-local LDS before SDOT and directly reduces experts. Its nearbyint/clamp
contract differs from grouped quantization. Simply enabling a grouped M32
consumer cannot produce another C1 fusion win without changing that contract.
The conditional C1 run therefore measures **isolation/non-regression** on the
unchanged native M1 graph, not a newly optimized C1 kernel. C1 A1 reuses the
completed C32 A2 control process; its files have an explicit state link.
C1 B/A2 use fresh processes. Completed C1 resident legs:

| Leg | tok/s |
|---|---:|
| A1 |89.8533|
| B1 |90.0337|
| B2 |90.0144|
| A2 |89.5909|

Control mean89.7221, candidate mean90.0241 (+0.337%). This is treated as
small process/time variation, NOT a C1 optimization: M1 does not select the
candidate. All eight1453-token completions pass ID-to-text checks; each leg's
two complete ID sequences match. France passes all C1 sentinels. The fixed
single-case screen is not a replacement for the previous87.599 formal C1
multi-case result. Only the captured M1 graph executes during C1 residency.

All five owned service processes stopped normally; AMD-SMI subsequently reports
no running processes on GCD0--7. No production defaults changed. Retain the
tested native-M32 opt-in and archive the small positive result; do not start
another optimization or DSpark test from this request.

## Validation and artifacts

New native scope/actual-selector tests and existing DSpark isolation tests:
4 targeted tests pass. One other pre-existing DSpark test is stale and omits
`configured_blocks`; its full file does not pass. That unrelated test was not
silently fixed or counted as passing.

Artifacts: `../experiments/dsv4_tp8_ar_down_consumer_20260914/`, including
component.json, summary.json, all raw wave outputs, input manifest, service
logs/PID records, GPU owner records and scripts. Global/profile defaults are
unchanged; the new native-M32 consumer stays opt-in pending broader validation.

Final archive:92 verified raw JSON/log/patch files,6,810,993 bytes;
SHA256 `5ff5ff2648ab9ff4e763ff7690c9dc455e76bb022d8a7c62a3860966d8ec070e`.
Independent C1 analysis confirms one unique full completion sequence across
all eight measured waves, including different arms/processes.
