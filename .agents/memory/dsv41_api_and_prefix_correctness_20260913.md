# V4.1 API bring-up and prefix correctness, 2026-09-13

## Scope and status

Continues `c19d110f17` Q-norm / compact-down fixes, preserving the existing
uncommitted V4.1 fitting work. This is **not a clean-checkout acceptance or a
performance checkpoint**. Original checkpoint weights, TP8/EP1/no-A2A,
pinned private host Engram, eager execution, FP8 KV, chunk2304, actual
8192-token pool. No DSpark, no target approximation, no weight requantization.

The fitting goal remains open: the tests below establish useful text/API and
component behavior, not full-model bitwise correctness or broad model quality.

## New API defects and fixes

- `resolve_chat_encoding_spec` matched `DeepseekV41ForCausalLM` using the
  `DeepseekV4` substring and silently selected V4 encoding. V4.1 now dispatches
  first, including the offline simple-chat helper. Empty system messages are
  not injected into plain chat: V4.1 emits a real system marker for them.
- The bundled V4.1 encoder existed but was not connected to serving. Connected
  its tools, task, content-block and reasoning-budget path, while retaining
  the existing V4 path. Multimodal *inference* has not been tested here.
- This local checkpoint maps low/high/max to 50/75/100; another bundled V4.1
  revision maps low/high/xhigh/max to 25/50/75/100. Read local checkpoint
  mapping/default as validated AST literals (no checkpoint Python execution).
  Preserve bundled defaults when no validated local profile is available.
- Protocol now preserves integer reasoning budgets 1..100, including the
  nested reasoning.effort spelling, separately from other models' fractional
  budgets. Invalid V4.1 budget0.5 and protocol budget101 return HTTP400.
- Register `deepseekv41` function parser. Its existing class declared spaced
  tag names but inherited a V3.2 constructor with hardcoded unspaced patterns.
  Explicitly configure the V4.1 calls/invoke/parameter patterns, self-closing
  invoke, streaming detection and structural-tag interface. V4/V3.2 untouched.
- Add `deepseek-v41` reasoning-parser alias using the existing compatible
  think delimiters. The correctness launcher explicitly selects both parsers.

## Offline validation

Command (physical GPU4 is visible because existing import-time FP8 detection
probes the GPU even in these otherwise CPU-only unit tests):

```sh
HIP_VISIBLE_DEVICES=4 /home/pc/anaconda3/envs/DS/bin/python -m pytest -q \
  test/registered/unit/entrypoints/openai/test_dsv41_chat_encoding_contract.py \
  test/registered/unit/function_call/test_deepseekv41_detector.py
```

Result: **35 tests passed, 16 subtests passed**. Covers architecture dispatch,
budgets, literal-only local profile reading, XML and JSON calls, self-closing
and multiple invokes, reasoning/tool round trip, eight chunk sizes including
single-character streaming, required/named/auto grammar, old unspaced parser
contracts, and harness rejection of empty/truncated successful responses.
Existing unrelated broader serving tests require missing `datasets` through
test_utils; not reported as run/passing. The V4.1 detector test now uses plain
unittest.TestCase, since it does not need benchmark/test_utils functionality.

`check_dsv41_encoding.py` matches all **five checkpoint golden text fixtures
byte-for-byte**. Crucial harness correction: the checkpoint's own
test_encoding.py defaults fixture cases to chat, not thinking. The initial
checker used thinking and failed cases2/3/4; no encoder math was changed to
make these pass. Explicit per-case thinking still takes precedence.

## Fresh-process service/API validation

Validated no active API sockets, stopped only the old V4.1 test parent37473,
and started tmux `dsv41-api-20260913` using the foreground launcher. The user's
separate tmux `model` session was left alone.

- Parent58982, TP schedulers59625..59632.
- Log `/tmp/dsv41-api-20260913.log`; listening `0.0.0.0:30101`.
- Ready **11:26:27 HKT**, reported weight load103.31s, tokenizer startup136.06s.
- Actual `--tool-call-parser deepseekv41 --reasoning-parser deepseek-v41`.
- No forward-snapshot hooks in this new process.
- Post-load unique named GPU storage53.987GiB/rank; E8M0 routed scales2.930GiB;
  direct scale clones0; Engram large tables/scales in HBM0; pinned host
  mappings23.604GiB/rank. These are independent storage counts, not logical
  tensor sizes added across aliases. Engram lookup outputs still use GPU.
- Advertised model context1048576 is **not** the actual8192-token test pool.

`check_dsv41_chat_api.py` second full run: **10/10 checks passed**:

1. Non-streaming chat: Paris sentence, natural stop.
2. Streaming chat: same sentence, stop, usage and DONE received.
3. Integer budget75: separate nonempty reasoning + Paris answer, natural stop.
4. Auto lookup(city=Paris), non-streaming: parsed tool_calls with valid ID.
5. Synthetic tool-result round trip: `The temperature in Paris is 22°C.`
6. Auto lookup, streaming: reconstructed function/JSON arguments match.
7. Required tool call: correct calls block and parsed result.
8. Named tool call: correct calls block and parsed result.
9. Invalid budget101: HTTP400.
10. Invalid fractional V4.1 budget0.5: HTTP400.

**No external tool was executed**; the temperature is an explicit test fixture.
First run's last two checks were falsely failed by the checker accepting only
nested error/detail, while SGLang returned its valid flat object=error form.
The original raw responses and first summary are preserved; fixed the checker,
added schema tests, and reran all10. This was not a service failure.

After API tests, native `/generate` France returned the same eight IDs as the
previous process, fresh cached_tokens0 and natural EOS (1.717s request wall,
not a steady decode benchmark). C4 barrier-admitted varied France/arithmetic/
Python/SQL, two rounds, passed **8/8**, same per-case IDs across rounds and
matching the earlier pre-API C4 run. This is a small concurrency smoke, not
coverage of all scheduler boundaries or 32-way load.

## Earlier numerical diagnostics retained in this checkpoint

These were executed on preceding service37473, before the API-only restart.

- Short code continuation: cached vs full recompute at absolute prefixes
  68/69/70/127/128/129/255/256/257: **9/9 top1 equal**. Some logprobs differ;
  this is not a bitwise floating-point result.
- Real source-review prompt2376tokens crossed the2304+72 chunk split, producing
  384 coherent tokens without the previous collapse. It finished at length,
  not EOS, and contained a boundary-description mistake/typo. Do not call the
  answer fully correct or complete.
- Long-prefix cached vs full recompute at2376/2377/2378/2431/2432/2433/2559/
  2560/2561: **8/9 top1 equal**, divergence at2560, cached362 vs recompute343.
  Both have small top1 margins (0.25 vs0.125), with the same top20 set there.
- Three fresh full-recompute controls at each of2376 and2560:2560 produced
  top1 **[362,362,343]**;2376 top1 stayed15 but margin varied0.25/0.875/0.5.
  Thus full recomputation itself is not repeatable. This evidence **does not
  isolate a cache-addressing bug**. The first numerical divergence remains
  to be located; reduction order, selection and allocator effects are hypotheses,
  not established causes. Do not relabel this as full determinism achieved.
- Compact CK gate + HIP down M2048 real-weight component, tiled copies of17
  captured rows: stage1 relativeL2=.004341, output=.005404, cosine=.999989;
  three eager evaluations matched exactly. **No HIP graph replay was run at
  M2048** (report graph_replays=0). It exercises CK block-M switching, not
  diverse request routing or service throughput.

## Artifacts and reproducibility

Reports and raw API responses:
`.agents/experiments/dsv41_api_correctness_20260913/`.
Larger raw prefix/long-review data remain under the explicitly named `/tmp`
paths in summary reports; preserve them before any reboot if doing first-div.

New tools:

- `scripts/rocm/check_dsv41_encoding.py`: official local golden comparison.
- `scripts/rocm/check_dsv41_chat_api.py`: repeatable JSON/SSE/API probes.
- `scripts/rocm/check_dsv41_cache_prefix.py`: fixed-prefix cached/recompute tests.
- `scripts/rocm/check_dsv41_concurrent_smoke.py`: four varied requests, two rounds.

Remaining: long-prefix first-divergence, independent full-model numerical
reference, broader code/reasoning quality, larger concurrency/context, graph
execution, and multimodal validation. `/v1/responses` is still disabled because
`openai_harmony` is missing; this turn verifies `/v1/chat/completions` only.
