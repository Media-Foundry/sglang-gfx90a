# V4.1 Q-normalization and compact W4A16 down correctness, 2026-09-13

## Scope and baseline

TP8/EP1/no-A2A, original `/media/PM983/deepseek-v4.1-flash`, pinned private
host Engram, FP8 KV, eager prefill/decode, 8192-token pool, chunk2304.
Baseline HEAD was `87e8376b6f` with the existing uncommitted V4.1 fitting
changes. This is not a clean-checkout performance claim. Source worktree
changes outside the numerical fixes below are preserved.

After the scheduled reboot, the first service reached readiness at 10:26:10
HKT. Proxy-free requests with explicit official chat token IDs completed but
returned gibberish. A second diagnostic process (PID27336, log
`/tmp/dsv41-firstdiv-20260913.log`) was ready at 10:34:06 and returned `ereg`
for a one-token France probe. The prefix has 17 actual tokens. Scheduler
logging says 256 new tokens due to bookkeeping; captured embedding input
is M17. Hook call1 is a run-ahead M1 decode, NOT an identical second prefill.

## Findings

1. The official V4.1 attention applies q_lora norm before wq_b, but no
   per-head Q norm afterward (`q_head_norm=False`). The shared HIP fused
   Q/K norm-RoPE-store helper always normalized Q. Added a `normalize_q`
   argument defaulting to True (unchanged V4 contract); both V4.1 callers
   now pass their model-specific `self.q_head_norm`.

2. V4.1 local expert intermediate is 288, padded to 384. The older AIter
   scale shuffler separately pads 12 K32 groups to 16. Its CKTile W2
   pipeline/scale indexing does not safely consume that combination.
   A small real-checkpoint oracle isolates good stage1 and grossly bad
   stage2. The log's printed legacy CK kernel names do NOT identify the
   actual executed backend: `get_2stage_cfgs` prioritizes CKTile for
   Dsv4Silu BF16/per1x32 before testing those names.

The captured layer0 rank2 routed output RMS is 26.55 versus the normalized
input RMS 0.125. Adding captured routed and shared partials across all eight
ranks closely reconstructs the actual collective output (relative L2
0.001687), pointing upstream of the collective. The raw-checkpoint W4A16
last-token reference has output RMS 0.0178--0.0419; the old output is about
55--67 times that norm with low cosine similarity.

## Real-weight component oracle

Eight fixture experts hold six actually selected layer0 experts
`[282,10,369,355,118,167]`, using captured inputs and real nonconstant E8M0
scales. Only the fixture is loaded on physical GPU4; checkpoint bytes are
unchanged. Stage1 uses actual CKTile, not a mock.

| Rank0 M1 configuration | Stage1 relative L2 | Full output relative L2 | cosine | replay exact |
|---|---:|---:|---:|---|
| Current K384, scale16 | 0.004304 | 74.9009 | 0.0900 | no |
| K384 + old W2 row compensation | 0.004304 | 75.2975 | 0.5521 | no |
| K512 + W2 row compensation | 0.004304 | 0.015077 | 0.999983 | no |
| K384 + compact HIP down | 0.004304 | 0.005082 | 0.999990 | yes |

The independent reference expands only selected FP4 weights and uses BF16
GEMM. It is not a bit-exact target implementation. Small CK accumulator /
SwiGLU rounding differences remain. No speedup is claimed.

## Implemented compact down

Keep CKTile gate/up and its 384-wide intermediate. A new wave64 HIP down
consumer independently addresses the A16W4-v1 weight shuffle (K384) and
scale shuffle (16 groups), reads only the 288 logical values, applies route
weight before down with BF16 rounding, writes per-original-slot FP32
partials, then rounds each expert projection to BF16 and sums slots in
fixed order. There are no atomic output sums or HBM-resident expanded
weight copies. Scratch is M*6*5120*4 bytes, e.g. about 1.99 MiB for M17.
This is a correctness-first implementation, not a tuned throughput kernel.

The selector uses an explicit loader layout marker carried through the
FP4 Tensor view in `get_aiter_quant_info`. It is restricted to gfx90a,
V4.1 TP8 E384/H5120/I288-padded384, legacy A16W4-v1 shuffle, no EP dispatch,
no no-combine, and bounded SwiGLU=10. Existing V4 paths do not enter it.

Rank2 M17 real-input component: full relative L2 0.006322, cosine 0.999987;
all 1000 HIP graph replays were checked on-device against the first replay
and matched bitwise. Rank0 M17 also passed, but its earlier graph probe
checked only the final state after 1000 replays; do not present that earlier
probe as per-replay validation. JSON reports live in
`.agents/experiments/dsv41_correctness_20260913/`.

Q/RoPE tests cover M1/3/17, H8/H16, varied positions; non-RoPE Q stays exact,
RoPE relative L2 <=0.000568, default normalized V4 path unchanged, and KV
stores exact between flags. Dense FP8 block32 real-weight tests separately
passed at M1/3/17/32 (relative L2 about 0.0016).

## Bring-up tools

- `scripts/rocm/start_dsv41_correctness.sh`: check port, amd-smi processes,
  checkpoint/Engram/syntax consistency, then foreground serve; never kills
  another process, never equates launch with readiness.
- `check_dsv41_http.py`: bypass proxies, pass explicit token IDs, require
  nonempty completion-only IDs matching reported count and response text,
  retain original JSON, reject overwriting a previous trial.
- `dsv41_trace_hooks.py`: existing forward-hook API, bounded eager snapshots,
  no replacement outputs, sampling or additional collective.
- `check_dsv41_cktile_contract.py`: real selected-expert layout oracle.

## End-to-end status

New process PID37473 started at 10:51:58 with Q fix and compact down enabled
by the explicit loader marker. It uses `/tmp/sglang_dsv4_flash_ar.log` (the
first invocation of the new wrapper used launcher `start`; the wrapper was
then corrected to `serve` for future foreground/tmux use).

At 10:54:24 the new service became ready on `0.0.0.0:30101`. All eight
ranks logged the actual compact-CK selector hit on the first request.
Post-load audit still reports GPU Engram 0.000 GiB, host mmap 23.604 GiB per
rank, unique named GPU storage 53.987 GiB. No extra resident weight copy was
introduced by the down repair.

End-to-end checks now pass:

- Two fresh-cache official France requests returned exactly
  `The capital of France is Paris.` and stopped at EOS. Both returned the
  same eight IDs, SHA256
  `ca5a08a13ccbe69da69496f9b8649118da1cf3b5a0c6840121f7ad16856ded75`.
  Cold request wall time (including hooks/JIT) was 10.918 s; warm 1.729 s.
- A real dependency-code prompt (68 input tokens) produced 248 tokens and
  natural EOS in 47.875 s. The answer is coherent Python implementing
  Kahn topological sort. After reading the code, its function was run with
  restricted builtins: four empty/DAG/prerequisite-only/duplicate-edge
  cases and two cycle cases passed. It scans all nodes per pop, so it is
  O(V^2+E), not an optimized O(V+E) implementation. This is a semantic smoke,
  not a broad coding benchmark. The 248 tokens cross the 128-token SWA
  boundary without the earlier visible collapse.
- `/v1/chat/completions` with normal messages, temperature0, max32 and
  `chat_template_kwargs={"thinking":false}` returned the same France
  sentence with `finish_reason=stop`, 17 prompt / 8 completion tokens.
- Captured actual service layer0 routed partials now match raw checkpoint
  W4A16 references on **all eight ranks**: relative L2 0.00305--0.00604,
  cosine >=0.999985. The prior 55--67x norm explosion is gone.

Remaining validation: long prefill/compressor boundaries, concurrency,
teacher-forced logits against an independent full-model reference, and
broader quality. The advertised model maximum is 1M but the actual test
pool is only 8192 tokens. Current eager speed is not a performance checkpoint
(the code request averages about 5.18 completion tok/s including prefill).
General model correctness is not proven by these smokes.

API follow-up: current `chat_encoding.resolve_chat_encoding_spec` selects
`dsv4` for the V4.1 architecture substring. Basic text chat succeeds, but
the repository's separate `encoding_dsv41.py` is not selected there; audit
tools/thinking/multimodal differences before claiming complete API support.
