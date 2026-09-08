# Final HC-head geometry screen

After user withdrawal of down-prefetch, inspected current production source and
prior MHC negative experiments. This is final HC head, not per-layer MHC or
Qwen hc_combine. It executes once per forward, so do not multiply savings by43.
Production calls fused_hc_head for nonempty input; K512/D512/four warps, one CTA
per token, with sequential K16384 scan. No production edits in this screen.

amd-smi reported all eight GPUs idle before the isolated test. Physical GPU4
only, no service or model loaded, synthetic BF16 x and FP32 fn/scale/base.
Five ABBA cycles,200 graph replays per leg,20warmup;100 input/weight/gate
mutations and1000 repeat checks per shape. Adjacent JSON preserves samples.

|M|warps/D|A us|B us|exact mutations|
|---|---|---|---|---|
|1|4/1024|58.039|44.088|72/100|
|32|4/1024|59.441|45.226|0/100|
|32|8/512|59.331|54.578|0/100|
|32|8/1024|59.363|61.688|0/100|

Self-control4/512 is100/100 exact at both M. All variants stable1000/1000;
max BF16 absolute delta up to0.015625. Merely fixing K512 does not guarantee
compiler reduction/layout arithmetic is unchanged when D or warps changes.
This is deterministic numerical variation, not evidence of runtime race.

Do not enable a production selector: best~14us/forward saving is only~0.12%
of C1 at84tok/s or~0.05% of C32 at~980 aggregate, assuming fully critical.
These are ceilings using measured local savings, not E2E results. Synthetic
parity failures do not prove bad answers, but no teacher-forced/model oracle
has been run for this candidate. Current production is untouched.
Next HC-head work, if pursued, must preserve explicit reduction ownership and
beat the whole final-head chain; no reason to change runtime defaults here.
