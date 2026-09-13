# V4.1 two-level indexer candidate blocks: HIP wiring

Parent checkpoint: `50fae816fa` (cutoff stability + previously tested fitting).
This is correctness work on original weights, not a performance experiment.

## Missing semantics and implementation

The local official `inference/model.py` selects candidate blocks at layer20,
then constrains index sources24/28/32/36 to those blocks. Each consuming layer
still computes its own scores and Top-512. The HIP backend previously ignored
both `is_candidate_source` and `uses_candidates` entirely.

Actual checkpoint: candidate budget2048, block size8, ratio1 at source and
consumers. Pruning can first matter above16384 positions. Config SHA256:
`8be45ce0476004a3f529fd896115a4a2e800a129ad2d3ec05b16050f52e21879`.

Implementation keeps the reference's block maximum, newest-block pinning,
causal -inf treatment and omission of unreachable filler blocks. Equal block
scores use increasing logical block ID to avoid the same nondeterministic
cutoff problem fixed in position Top-K. No score perturbation or weight edit.

State is attached to one forward's attention metadata and keyed by request
ID. A source overwrites it; every metadata init/copy clears it. Consumers fail
closed if source/ratio/shape is missing or incompatible, rather than using an
old request's candidates. Only block-level bool masks remain live across
layers (approximately1/8 of a full position mask). Consumer masks are applied
after causal masking; selected -inf filler positions are emitted as -1.

## Verified so far

- New unit tests cover widths0/1/7/8/9/17/16384/16385/32769, budgets1/2/2048,
  ties, empty rows, pinned weak newest blocks, and partial blocks. The oracle
  is an independent scalar loop, not the production tensor selection.
- Actual HIP-backend method invoked with CPU fake caches checks ragged
  source/consumer wiring, full source Top-K, restricted consumer Top-K,
  nontrivial logical-to-physical mapping, missing-source errors and metadata
  copy/reset ownership. Existing DSpark SWA replay tests also pass.
- Combined bringup/API/parser/candidate/SWA unit suite:76 passed plus16
  subtests. Initial collection failed because the plain shell lacked the
  launcher's local `sgl_kernel` PYTHONPATH; setting AIter alone did not fix it.
  Successful invocation used the three paths below. No service crash involved.
- GPU4 component compares against the pure `select_candidate_blocks` function
  extracted by AST from the local official file, without importing/loading
  that whole model. Widths16383/16384/16385/32769 all match masks exactly;
  each passes1000 HIP graph replays plus10 changed-score/changed-position
  replays. Unique reachable scores isolate reference semantics from its own
  unstable ties. Tied ordering is separately covered by the scalar oracle.
  Evidence: `../experiments/dsv41_candidates_20260913/component.json`.
- Helper graph replay is NOT a claim of full-service graph support.

Successful unit environment:

```bash
export PYTHONPATH=/home/pc/Code/sglang/python:/home/pc/Code/sglang/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:/home/pc/Code/sglang/python/sglang/kernels/aot/python
export SGLANG_USE_AITER=1 HIP_VISIBLE_DEVICES=4
/home/pc/anaconda3/envs/DS/bin/python -m pytest -q \
  test/registered/unit/test_deepseek_v41_bringup.py \
  test/registered/unit/entrypoints/openai/test_dsv41_chat_encoding_contract.py \
  test/registered/unit/function_call/test_deepseekv41_detector.py \
  test/registered/unit/layers/attention/test_dsv41_candidate_blocks.py \
  test/registered/unit/layers/attention/test_dspark_swa_loc_replay.py
```

## Service regression and remaining gates

Test service restarted via `scripts/rocm/start_dsv41_correctness.sh`, tmux
`dsv41-candidates-20260913`, PID98984, log
`/tmp/dsv41-candidates-20260913.log`. Same TP8/EP1/eager/8192 pool/chunk2304
profile, original weights, host Engram, no tensor-summary hooks. Ready at
12:13:15 HKT; startup log reports103.96s weight loading and137.18s tokenizer
startup. The following real-weight tests are complete:

- France returns `The capital of France is Paris.`, natural EOS, same8 IDs.
  First request after restart took9.395s; not comparable to the previous warm
  1.714s as a performance measurement.
- All nine fixed prefixes2376/2377/2378/2431/2432/2433/2559/2560/2561 have
  identical next-token IDs, returned token logprobs and top-20 logprobs to
  the pre-candidate checkpoint's FULL-RECOMPUTE responses. This is distinct
  from cached-vs-recomputed logprobs, which still differ as previously noted.
- New fresh-cache192-token source-review generation: every token ID,
  token-logprob row and top-20-logprob row matches the earlier checkpoint.
  Both generations end at the requested192-token limit, not natural EOS.
- C4 short semantic probes:2initial +3additional warm rounds,20/20 valid
  answers and natural EOS. France/arithmetic/Python IDs repeat. SQL switches
  between bare `SELECT COUNT(*) FROM events;` and the same query in a Markdown
  code fence, including during warm repeats. Thus dynamic concurrent output
  is NOT bitwise stable. Logs show different admission groupings (1+3 vs2+2),
  but this is not a controlled attribution experiment. Do not label it fixed
  merely because both answers are semantically correct or because C1 repeats.

Evidence is committed under `../experiments/dsv41_candidates_20260913/`.
The two serving source file SHA256s were captured while PID98984 was loading:

- `dsv41_sparse.py`:
  `32b373de99b19d6fea41c59b0dcab5c62a1d44fe2d735205456e5caa5f31c8fe`
- `deepseek_v4_backend_hip_radix.py`:
  `c0c2e5696f13e8537cec4545595ec5fc428d69db1be2755b456e6aa68c94bdce`

Storage audit remains unique named GPU53.987GiB/rank, E8M0 routed
scales2.930GiB/rank, host Engram23.604GiB/rank, gpu_engram0 and scale clones0.
Those counts do not include transient attention workspace. No V4 throughput,
long-context full-model correctness or general quality claim is made here.

Full-model >16K testing still needs a larger pool AND bounded indexer score
workspace. The eager score path materializes [queries,32,positions] BF16;
M2304/L32768 is about4.5GiB for that tensor alone, before ReLU/multiply
temporaries. The log's~2.54GiB post-pool free memory is not an allocator peak
budget: cached reserved storage may also be reusable. Measure both allocator
and physical headroom before raising context; do not assume this tensor fits
or that the free-memory log proves it cannot. Query tiling is a separate next
change and needs numerical/Top-K/E2E comparison; do not silently change chunk
size and call it the same experiment.
