# Route producer longer-input regression

OriginalV4/TP8/EP1/nativeAR/originalweights/1Mpool/32Kchunk,C16 heterogeneous
public-code requests,zero prefix cache. Both arms retain owner-K32 and common
FP32/20 MHC; only route producer changes. Three scored waves perABBA leg.

## 16K accepted

| Leg | Median input tok/s |
|---|---:|
|A1|10132.673885|
|B1|10214.171757|
|B2|10219.617908|
|A2|10136.473024|

Control10134.573454,candidate10216.894832,+0.812283%.Control drift0.037494%,
largest within-leg range0.099763% (a screening rule,not confidence interval).
192 full128-token outputs agree across waves/processes;1008 teacher positions
have exact logprobs and Top5 records versus both controls and prior accepted
K32. All8 ranks selected route producer in B only,all retained K32/1Mpool.
Owned16K services stopped. No new live reference checks:1376 were prior8K.

## 32K timing complete; acceptance stopped on teacher-history drift

All32K services stopped. A1/B1/B2/A2 medians are
9843.052229/9934.898708/9933.363491/9844.204755,but session29781 exited1 in
analysis. Current A1/A2/B teacher inputs and1008 records match; the historical
comparison used a DIFFERENT continuation. `analyze_v1.py` and original run.log
preserve the failed analysis; teacher-lineage.json audits every request.

Candidate bridge on the actual historical teacher prompts then found real
historical drift(max logprob delta1.266876,979/1008Top1 same). Requests0..9
differ while10..15 match. Fresh control replay completed and matches the fresh
candidate at ALL1008 logprobs/Top5 positions. Both differ from history; this
paired check excludes route-producer as the cause. All GPU processes stopped.
Do not claim32K acceptance; no scored wave was replaced. Prior accepted32K
remains9854.324387. Revised acceptance requires a successful same-input bridge.
Pending reducer/profile-consolidation GPU experiments are paused during this
investigation. Runtime code and global defaults remain unchanged. The bounded
investigation archive includes both failures,all scored waves and both fresh
teacher replays; it is explicitly NOT a32K acceptance artifact.
