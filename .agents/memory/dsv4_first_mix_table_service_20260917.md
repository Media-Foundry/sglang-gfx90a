# First-MHC vocabulary table: live exact and +0.45% C16x8K service

Continuation of e00e9f9b5e. Original DeepSeek-V4-Flash, original checkpoint,
TP8/EP1/no-A2A, native AR, 1M logical KV pool, 32K chunk, 16 heterogeneous
public-code inputs, 131069 input tokens/wave, zero prefix cache hits.

## Implementation / contract

Default-off SGLANG_DSV4_PREFILL_FIRST_MIX_TABLE derives only the first-layer
24 FP32 raw mix values/token from static embedding + original FP32 Fn/epsilon.
Every rank builds its own 16160 vocabulary rows in 1024-row chunks, then native
TP all-gather assembles [129280,1,24]. Persistent size 12410880 bytes (~11.84MiB)
per GCD; repeated embedding temporary <=32MiB. No full embedding replica, no
checkpoint mutation. Original Sinkhorn, weighted mixing, norm, later layers
and decode are unchanged.

Model-owned cache records tensor identities/pointers/versions, keeps weights
alive without copying, and establishes stream readiness via an event. Loader,
post-loader and embed/head replacement invalidate it. Online updater APIs reject
this experimental static-checkpoint profile; arbitrary .data.copy_ cannot be
assumed to increment a version. Unexpected cache-key changes fail rather than
starting a rank-local collective rebuild.

Scope requires original V4, TP8, plain native EXTEND, first layer, 43 layers,
COMMON_FP32 + COMMON_UNFUSED policy (small rows need COMMON_SMALL), native
FP32 backend, no capture/input_embeds/LoRA/real multimodal input. Pure text often
has mm_inputs=[None,...]; exclude actual non-None multimodal entries, not the
list itself. V4.1, AR decode, DSpark and draft remain outside scope. Global
launcher defaults unchanged. B/start-ar-matrix.sh is the measured opt-in setup.

## Verification and formal ABBA

Experiment: .agents/experiments/dsv4_first_mix_table_service_20260917/.
Diagnostic check completed 152 actual first-MHC calls, balanced across all 8
ranks, covering small and large M. Every lookup byte-equals the existing FP32
kernel on the actual residual; separate consumption checks prove selector hit.
Eight tables built once each. Diagnostic throughput is NOT a performance result.

Formal noninstrumented runs: A1/B/A2 fresh processes; B contains consecutive
B1 and B2 legs, 3 waves per leg. Same corrected FP32 reference in both arms,
route producer enabled, checks disabled, unscored warmup excluded.

| Leg | Median input tok/s | Wave range |
| --- | ---: | ---: |
| A1 | 10322.223021 | 0.2624% |
| B1 | 10390.892735 | 0.1625% |
| B2 | 10364.352511 | 0.0458% |
| A2 | 10340.090734 | 0.1755% |

Control center 10331.156878; candidate 10377.622623: +0.449763%.
A1->A2 control drift +0.173099%. This is a small measured improvement, not a
large model speedup or an exhaustive statistical confidence claim. Each rate
was recomputed from earliest request start to latest first-token arrival.

Three processes x two 128-token output waves: all 16 answers identical across
all six waves. Fixed-continuation teacher forcing: A1/A2 and A1/B both exact
at all 1008 non-null selected logprobs AND all Top-5 lists. Candidate output
also equals the previous corrected-reference B output for all 16 requests.
France passed. This verifies the tested execution conditions only, not universal
batch invariance or correctness of every generated code claim.

accept.py passed. All owned services stopped; amd-smi afterward showed no live
GPU processes. CPU tests previously passed 16 cases (one unrelated config
warning); do NOT run them concurrently with a service, see incident below.

## Failures preserved, not counted

1. Initial diagnostic check silently skipped because pure-text mm_inputs was a
list of None. Mandatory-hit assertion rejected the run. Original sources and
outputs frozen in no-hit-evidence.tar.gz + manifest; check-no-hit retained.
The predicate was corrected before the successful 152-check run.
2. First formal B aborted in ready() exclusivity check: independently launched
pytest PID3035562 appeared as a GPU owner even with HIP_VISIBLE_DEVICES=''. No
timed leg/progress file existed. This was our test orchestration error, not a
model failure. Preserved as B-owner-guard and B-owner-guard-driver.log. resume.py
validates successful A1, empty owned-stop states and unchanged runtime hashes,
then reruns only B/A2 after all GPU owners cleared. Never count the failed B.

## Remaining work

Longer-length/mixed-prefix coverage remains pending; do not relabel previous
16K/32K numbers as revalidated with this cache. Keep 1M pool and default-off scope.
Whole-model drift still has a subsequent QKV projection shape/row-order source
(see dsv4_unfused_mhc_qkv_contract_20260917.md). This table preserves the corrected
first-MHC math; it does not fix that separate projection. Do not repeat the
already slower fixed-K Triton geometry sweep or claim ordinary rocBLAS solves
all QKV cross-M differences.
