# Exact direct-row CK dequant: 10.206k TP8 C16 prefill, +2.31%

2026-09-16, base1224b7ff6c. This turn made progress: real stage1 counters,
three bounded producer designs, exact runtime integration and formal serviceABBA.
Original V4 Flash, TP8/EP1/no-A2A/native AR, original checkpoint precision,
1048576logicalKV,C16x8K diverse real code, zero prefix hits,32Kchunk,
131069inputtokens/wave. No DSpark or V4.1 measurement.

## Formal service result

| Leg | Median input tok/s |
|---|---:|
|A1 accepted owner checkpoint|9982.643457|
|B1 direct-row expansion|10212.478790|
|B2 direct-row expansion|10199.322842|
|A2 fresh owner control|9967.391617|

Three scored waves/leg, warmup excluded. Fresh A1/B/A2 processes; consecutive
B1/B2 legs in one candidate process. Control9975.017537, candidate10205.900816,
**+2.314615%**. Control drift-0.152784%. All six candidate waves exceed10000;
lowest10193.643163. This is a measured result for this workload, not a universal
10k throughput guarantee. No concurrent GPU benchmark or clock changes.

All prior exact-owner/H16/local-sink/uniqueCK/vec4/exactpost settings unchanged.
Owner-only indexer query producer and cooperative MFMA remain OFF.
No new persistent buffer, replica, checkpoint conversion or KV reduction.
Global defaults remain unchanged; explicit measured launcher is
`experiments/dsv4_dequant_direct_20260916/validated-launcher.sh`.

## New dataflow, not another dequant grid sweep

Existing `run_shuffled` decoded512values/tile through BF16 LDS, a separate
scale tile and three CTA barriers. New kernel reads one complete packed K32
row per lane with a16byte vector load, reuses its scale, and writes four
16byte BF16 vectors directly into the SAME [klane4,nlane16,kpack8] CK tile.
Four wave64s process independent tiles; no shared memory or cross-wave sync.
Original FP4 integer-code conversion, E8M0 handling, FP32 multiplication and
BF16 cast remain unchanged. The separate bit-construction candidate is NOT used.

Production grids:gate/up1664blocks;down416blocks. Actual dependency-matching
runtime compilation shows16byte loads/stores, no s_barrier, no LDS instructions,
0LDS/0private scratch; gate62VGPR/65SGPR,down63VGPR/66SGPR,wave64.
Static instruction counts are not runtime utilization. Compiler/source/module
hashes, metadata and disassembly are archived.

`SGLANG_DSV4_DEBUG_CK_DIRECT_DEQUANT=1` is default-off. Parent shape validation
plus `mix_pair_active()` restrict it to original-V4 TP8/EP1 ordinary large
EXTEND,I256,M8192..36864,logical packed FP4 and shuffled BF16 output. Requires
normal weight expansion, not raw-B CK or keep-expanded-weight diagnostics.
Native decode,draft,target-verify,V4.1,TP4,CP/TBO,capture and unsupported shapes
retain current paths. Existing scale-layout inverse is unchanged.

## Bounded screens

Earlier direct-output8-values/lane removed LDS but left W13 flat:~3.15ms.
W2 improved1.33->0.90ms. Direct BF16 bit construction for normal exponent ranges
improved W2 to~0.81ms but W13 remained flat; not adopted. These exploratory
JSONs predate the final extended prototype header; final accepted evidence is
rows-screen.json and the service artifact/source hashes.

Whole K32 vector-row version:

| Tensor | Production ms | Candidate ms | Grid |
|---|---:|---:|---:|
|W13 TP8 [256,512,4096]|3.10988|1.73916|1664|
|W2 TP8 [256,4096,256]|1.34240|0.84250|416|

These are individual expansion component graphABBA medians, NOT routed/E2E
throughput. Both emit original-size BF16 workspace; no online packed CK loader
or full-model expanded-weight cache was introduced.

All16FP4 codes tested under all256E8M0 byte encodings against the existing
producer, including its exceptional/subnormal behavior. This proves reference
equivalence, not an independent format-spec oracle. Historical actual weights
and varied shuffled scales reconstruct captured BF16 CK weights byte-for-byte.
Ten private input/scale mutations, poisoned output and100graph replays pass
per tested configuration. Real checkpoint files are never mutated.

## Integrated model correctness

Independent check service runs reference expansion alongside candidate:
43layers x4forwards x8ranks x2weights = **2752whole-weight byte comparisons**,
all pass. No sampled-only comparison. Diagnostic5638.36inputtok/s includes
duplicate expansion, temporary reference buffers and synchronizing equality
checks; it is explicitly NOT a performance result.

All four services pass France; C1 short prompt does not hit the new selector.
Timed arms each run4C16x128-token quality waves:192complete responses total.
All16requests match across all12waves and both settings. Prompt echoes, zero
cache hits, output-ID lengths and decoded text are checked, not empty arrays.

Fixed64-token reference continuation appended to each prompt: the leading
API null is validated and excluded, leaving16x63=1008positions. A1/A2 andA1/B
both show zero selected-token logprob delta,1008/1008Top1 agreement and
1008/1008identical returnedTop5 records. Not full-vocabulary logits and not a
universal batch/length/prefix determinism guarantee; no broad quality claim.

Eight CPU scope/negative-dispatch tests pass; syntax/whitespace checks pass.
All four owned service trees stopped and amd-smi shows no remaining GPU owners.
Runtime memory pickle stays unstaged. No installed AIter/CK module modified.

## Evidence and next step

`experiments/dsv4_dequant_direct_20260916/`: finalprototype, exhaustive/screens,
runtime source, service.py --arm check/A1/B/A2, analyze.py, summary.json,
validated-launcher.sh and service/compiler archive with manifest.
Stage1 counters are separately recorded in `dsv4_ck_stage1_counters_20260916.md`.

Retain exact direct-row path as explicit best configuration. Broader coverage
and fresh profile may follow; do not reuse the old dequant budget or add component
percentages to the already-measured2.31%service gain. CK N-stripe remains rejected.
