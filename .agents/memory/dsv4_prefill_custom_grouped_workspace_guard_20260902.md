# DSV4 custom grouped-prefill workspace guard (2026-09-02)

## Regression found

After lifting the compressor plan above 64K rows, a 32-request real-code
prefill containing 73,724 prompt tokens could reach the custom gfx90a routed
path.  Both the MFMA64 down kernel and the grouped-SDOT fallback materialize an
FP32 `[M, topk, hidden]` partial.  At M=73,724, top-k=6 and hidden=4096 this is
about 6.33 GiB per GCD.  With the 131K KV pool the allocation failed and killed
the request.

Guarding only `use_mfma32_prefill` was insufficient: the selector then entered
`gfx90a_fp4_expert_down_grouped`, which owns the same partial workspace.

## Fix

Add `SGLANG_DSV4_GFX90A_FP4_MFMA_PREFILL_MAX_ROWS`, default 16,384, and apply it
to the complete `use_grouped_prefill` selector.  M=2304 production prefill and
all decode tiers remain unchanged.  Larger prefill batches use AIter until the
custom down consumer becomes tiled or fused and no longer needs O(M) partials.

## Validation

- Python compile, shell syntax and `git diff --check` passed.
- GPU tests used physical GCDs 4,5,6,7 after `amd-smi process` showed no running
  workloads.
- The same 32 distinct real code requests (73,724 prompt tokens) completed with
  one output token each instead of OOM.
- The OpenAI chat endpoint answered: `The capital of France is Paris.`
- The diagnostic service was stopped; `amd-smi process` reported no remaining
  processes.

The recorder-enabled oversized AIter run took 185.69 seconds (397.03 aggregate
input tok/s).  This is a safety/correctness result, not a performance result.
It also shows that production chunk planning should not coalesce 73K rows into
one forward.  No new recorder file was written on the forced PTY shutdown, so
no occupancy claims are made from this run.

