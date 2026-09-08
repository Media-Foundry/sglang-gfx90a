# Fresh-process fixed-first-use TP8 down ABBA

Completed after `ed3c4dd873`, 2026-09-08. Raw audited aggregates and artifact
SHA256s are in the adjacent JSON. This is four independent service launches,
not the earlier paired-graph same-process experiment.

## Results

| Arm | PID | C1 trimmed | C32 HTTP warm | C32 resident warm | C32 cross-round exact |
|---|---:|---:|---:|---:|---:|
| A1 uniform | 3564536 | 84.183162 | 990.139833 | 1036.124244 | 10/32 |
| B1 baseline | 3573304 | 83.858418 | 985.101551 | 1031.276554 | 10/32 |
| B2 baseline | 3581829 | 83.268888 | 980.511374 | 1027.020153 | 3/32 |
| A2 uniform | 3589516 | 83.413359 | 986.486947 | 1034.427640 | 6/32 |

Arm geometric means, native tokens/s:
- C1: 83.563133 -> 83.797377, **+0.2803%**.
- HTTP C32: 982.803783 -> 988.311702, **+0.5604%**.
- Resident C32: 1029.146153 -> 1035.275594, **+0.5956%**.

C1 is three fixed real code prompts, four measured rounds, trim min/max per
task then geometric mean. C32 is the fixed 32 diverse-code corpus, six waves
of 256 outputs; drop wave zero then median, followed by arm geometric mean.
No speculative/accepted-token measurement. Do not confuse resident with HTTP.

## Correctness and capacity

All 48 measured C1 complete token sequences match the common reference across
arms; recomputed uint32 hashes pass. Each process passes France 32/32 through
reference EOS. All 768 C32 completions have length 256, finish=length and
recomputed completion hashes. C32 is **not bitwise deterministic** (table);
these checks do not establish broad semantic correctness or generated-code
execution success. The down-kernel component's earlier exact mutation/graph
oracle remains distinct evidence from these service checks.

All four logs have eight unique rank markers: 460 MiB ordinary baseline
warmup allocation, 2 MiB alternate-module increment per GCD. Graph capture
remains approximately 0.65 GB/GCD; available memory 15.58–15.64 GB. Every
process reports max_total_num_tokens=context_len=1048576. No extra graph or
weight cache. This verifies allocated pool capacity, not a live 1M request.

Private launch snapshots were compared without printing their contents:
identical cmd/cwd, only DOWN_UNIFORM and DOWN_FIXED_WARMUP environment keys
differ. No full environment snapshot is committed. No explicit fixed random
seed was added to the launch command; output equality checks are direct
evidence, not a claim that all runtime-generated state was identical.
AMD PID ownership audited before each arm; only exact owned services stopped.

## Interpretation and decision

The approximately 0.6% resident benefit agrees in direction/scale with the
earlier same-process paired ABBA (~0.53%). This fixed-first-use ABBA does not
show a C1 penalty. However, C1 declines across launches within both arms;
fixed module first-use does **not** eliminate process-to-process variability.
Neither a root-cause repair nor statistical significance is established.

Keep DOWN_UNIFORM and DOWN_FIXED_WARMUP default-off. Do not enable paired
graphs in production, shrink KV, or spend more kernel-rewrite effort on this
already-small metadata change. The bounded candidate is available for an
explicit opt-in profile; further promotion needs reproducibility/quality
evidence, not another isolated microbenchmark win.

Current owned service after this experiment: PID3589516, A2 single graph,
both uniform and fixed warmup enabled, paired graphs disabled, loopback30011.
No active benchmark controller remains at recording time.

## Audit tooling

`summarize_dsv4_paired_graph_abba.py --fixed-process-states A1 B1 B2 A2`
requires four different PIDs, completed single-arm states and correct ABBA
order; checks raw artifact hashes, common workload, C1 IDs, C32 completion
integrity and native statistics. It labels the result fresh-process, not
same-process. Six synthetic tests cover both modes and reject wrong arms,
reused PIDs, missing fixed-warmup flags, corrupt outputs and workload drift.
