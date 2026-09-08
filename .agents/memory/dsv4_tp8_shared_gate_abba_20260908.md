# TP8 shared gate fusion: completed service ABBA, no E2E benefit

Code checkpoint `db477056ff`. Only
`SGLANG_DSV4_GFX90A_TP8_C1_SHARED_GATE_ROUND` changes. Original weights,
native TP8/EP1/no-A2A; ROW_STABLE_PREFILL=1 fixed; pool1048576 fixed.
Eight-rank kernel-selection logs confirm both candidate services actually use
the candidate in BS1 capture. No added weight cache; graph memory remains0.65GB
per GCD, rank0 available15.64GB after capture.

| Block | Flag | C1 tok/s | C32 warm E2E | Warm resident |
|---|---:|---:|---:|---:|
|A0|1|83.96581|982.26155|1028.75076|
|B1|0|84.25399|981.73523|1027.19086|
|B2|0|84.13352|982.51780|1027.73699|
|A3|1|84.02667|981.18677|1027.18103|

C1 is the geometric mean of three code-case medians (two measured256-token
requests per case, excluding warmup). C32 uses the median of five warm waves,
discarding wave0 from six waves,32 distinct coding prompts per wave. B1/B2
reuse the same control process; A0/A3 are separate processes. Do not describe
this as four independent processes or claim statistical confidence.

Geometric means of block rates:
- C1: baseline84.19373, candidate83.99623, -0.23458%.
- C32 E2E: baseline982.12644, candidate981.72401, -0.04097%.
- C32 resident: baseline1027.46389, candidate1027.96560, +0.04883%.

## Correctness

All24 measured C1 full sequences match the existing reference, France passes
every block. All24 teacher-forced next IDs/input-logprobs/top20-logprobs match
across the four blocks. All768 C32 requests have256 completion IDs, finish
length, and matching saved SHA256 values. Dynamic C32 cross-round exactness is
6/32,6/32,8/32,6/32 by block: full batch determinism remains unresolved.
These results do not prove correctness for all prompts, positions or tiers.

## Decision

No E2E win despite standalone~51% component reduction. The timings are
consistent with the shared work being hidden by routed work, but no new trace
was taken to prove that attribution. Keep the verified experiment default-off;
restore serving flag off. Do not spend another geometry sweep on this path.
No changes to other validated optimization flags, precision or KV capacity.

Artifacts: `/tmp/dsv4_tp8_shared_gate_abba_20260908.json` and referenced per-block
C1/C32 files. Portable summary in adjacent `_summary.json`; reproducible
validation via `scripts/rocm/summarize_dsv4_shared_gate_abba.py` (requires a
complete sequence, validates IDs/digests and fixed workload before scoring).

Next decode work should use a fresh current-profile C32 critical-path trace
before selecting a new structural candidate. Earlier profile predates M32
attention overlap/legacy AR/gate-prefetch and cannot establish today's precise
stage budgets. Do not resume withdrawn attention-prepare/down-prefetch work.
