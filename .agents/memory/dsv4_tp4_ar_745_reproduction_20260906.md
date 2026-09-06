# Historical TP4 native-AR 74.5 tok/s reproduced (2026-09-06)

## Outcome

The August result is reproducible on four MI250 GCDs, TP4/EP1/no-A2A,
without speculative decoding. It was NOT a TP8 or DSpark measurement.
Original pre-rebase SGLang Python/JIT code achieves 74.018 HTTP tok/s median
over seven 256-token requests, maximum 74.655; all seven completion hashes
match the historical `51e2ac132057ead3`.

Do not conflate a rebased commit's inherited experiment notes with an
experiment actually performed on that rebased tree:

| Tree | Author date | Commit date | Result today |
|---|---|---|---|
| `b00a4e11cf5de90af0ec50c3407aa8662a7ddf8f` | Aug 24 | Aug 24 | Historical speed and hash reproduced |
| `7953551a3017df6852d4a45740d13f8f7f01a1cf` | Aug 24 | Aug 27 | About 53 tok/s, different stable hash |

Both commits are titled `Increase gfx90a prefill chunk efficiency`.
Their harness scripts are identical, but their full trees are not.

## Isolation and environment

- Main repository HEAD was `6b4804b2f9`; existing dirty changes preserved.
- Separate detached worktrees: `/home/pc/Code/sglang-repro-7953551` and
  `/home/pc/Code/sglang-repro-b00a4e1`. No historical runtime code copied into
  production and no shared dependency reverted.
- Physical GCDs 4,5,6,7; AMD-SMI showed no processes before starting and
  between services. Test port 30011, loopback only.
- Current DS conda environment: Torch `2.12.0a0+git78258b9`, HIP `7.14.60850`.
- Both services used the current AOT `sgl_kernel` package and the same current
  external AIter build. This is not a complete historical dependency snapshot.
- AIter checkout HEAD `9a469a608`, dirty; custom all-reduce module SHA256
  `0854436bd273f015e3df4fc1ccd60858cd9ce5925dd0e8a03be43542326d6948`.
- Original checkpoint `/home/pc/models/modelscope`; no weight conversion.
- TP4/EP1, `moe_a2a_backend=none`, chunk2048, pool8192, memory fraction0.80,
  decode graph BS1 (dense/sparse dual graph). No DSpark profile or SBO.
- `/get_server_info` confirmed tp_size4, ep_size1,
  speculative_algorithm=None and speculative_num_steps=None.
- `serve` does not itself freeze GC. Invoked POST `/freeze_gc` once before
  the original-tree tests; also tested this separately on the rebased tree.

Reproduction command (replace both occurrences of the worktree to test the
rebased tree):

```bash
SGLANG_DIR=/home/pc/Code/sglang-repro-b00a4e1 \
PYTHONPATH=/home/pc/Code/sglang-repro-b00a4e1/python:/home/pc/Code/sglang/python/sglang/kernels/aot/build/lib.linux-x86_64-cpython-312:/home/pc/Code/sglang/python/sglang/kernels/aot/python \
HIP_VISIBLE_DEVICES=4,5,6,7 TP_SIZE=4 EP_SIZE=1 MOE_A2A_BACKEND=none \
PORT=30011 CUDA_GRAPH_MAX_BS_DECODE=1 \
/home/pc/Code/sglang-repro-b00a4e1/scripts/rocm_dsv4_flash.sh serve

curl --noproxy '*' -X POST http://127.0.0.1:30011/freeze_gc
PORT=30011 scripts/rocm_dsv4_flash.sh bench 256 7
```

## Measurements

Same historical 2+2 thinking-mode prompt, greedy, 256 completion tokens,
HTTP request wall time (not GPU-only decode). Every request ended at length.

| Tree / condition | Per-request HTTP tok/s | Completion hash prefix |
|---|---|---|
| Rebased, before freeze | 16.380, 53.273, 53.263, 52.659, 53.175 | `31038f036190a094` throughout |
| Rebased, after freeze | 52.530, 52.600, 51.127 | `31038f036190a094` throughout |
| Original, after freeze | 68.676, 71.308, 74.167, 74.018, 74.640, 74.655, 73.221 | `51e2ac132057ead3` throughout |

Original all-seven median74.018; last-five median74.167. Rebased first
request includes substantial cold work; exclude it from steady inference.
GC freezing did not recover the missing performance.
These sequential historical reproductions are NOT a randomized ABBA
single-variable production optimization claim.

### Real code prompts, C1

Three sequential rounds with fixed official input IDs from
`dsv4_tp8_diverse_32_input_ids.json`, requests03/15/31 selected by prompt
content (the prompt strings below are authoritative), fresh salts,
temperature0, ignore_eos=true, 256 completion tokens. No concurrency.

| Prompt | Three HTTP tok/s observations | Median | Completion SHA256 prefix |
|---|---|---:|---|
| Reverse a Python linked list | 17.922, 74.776, 74.797 | 74.776 | `bba331d6f1e0fe15` |
| SQL duplicate email query | 74.677, 74.753, 73.635 | 74.677 | `0727060b2e774f90` |
| Merge sorted arrays pseudocode | 18.200, 74.714, 74.252 | 74.252 | `25791ddcde7ab62a` |

Hashes here use SHA256 over little-endian uint32 completion IDs, 256 IDs
each; every case is stable across all three rounds. First linked-list and
merge requests have shape-cold overhead; do not present those as warm rates.
Outputs begin with relevant code/explanations, not repeating blank text.
France sentinel: both original and rebased trees pass 2/2 exact first-nine
IDs `[671,6102,294,8760,344,2619,51119,42499,1]`.
This is smoke correctness plus repeatability, not a proof of full model
numerical equivalence, executable code correctness or long-context quality.

## Concrete regression lead, not yet a production fix

Original `deepseek_v4.py::_is_fused_mhc_post_pre_enabled` explicitly allows
gfx90a whenever `SGLANG_OPT_FUSE_MHC_POST_PRE` is on. Its comment explains
that the small-token fused boundary does not require the standalone
TileLang pre/post kernels (whose MFMA lowering is unsupported on CDNA2).

The rebase moves the predicate into
`models/deepseek_common/amd/deepseek_v4_fused_mhc.py` and omits that gfx90a
exception: it requires standalone TileLang post/pre flags, or gfx95 AIter.
The historical harness enables FUSE_MHC_POST_PRE but disables standalone
TileLang pre/post. Current production still contains this restrictive
predicate. This is a concrete lost selector, not speculation about clocks.

There are also real boundary implementation changes: new
`apply_mhc_post_pre_boundary`, different norm handling and argument passing.
Therefore this two-tree test does not prove that restoring the predicate
alone fixes all performance/numerics. Next: isolate the gfx90a small-batch
boundary on current code, verify norm exactly once and preserve
fn_bf16/fn_fp16/global_batch_size contracts, then run France, fixed-input
logits and real-code E2E ABBA. Do not blindly enable unsupported standalone
TileLang kernels or remove all-reduce correctness barriers.

## Local artifacts

- `/tmp/dsv4_repro795_old_server.log`
- `/tmp/dsv4_repro795_france.json`
- `/tmp/dsv4_repro795_bench.log`
- `/tmp/dsv4_repro795_frozen_bench.log`
- `/tmp/dsv4_repro_b00_old_server.log`
- `/tmp/dsv4_repro_b00_france.json`
- `/tmp/dsv4_repro_b00_bench.log`
- `/tmp/dsv4_repro_b00_code.log` (full texts and output IDs)

Temporary logs are local; the tables above preserve essential measurements
in git. Isolated worktrees retained for the follow-up comparison.
Both test services stopped after measurement; final AMD-SMI reported no
running GPU processes.
