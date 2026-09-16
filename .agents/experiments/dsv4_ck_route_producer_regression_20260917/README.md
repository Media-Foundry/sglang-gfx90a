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

## 32K pending

The same sequential driver(session29781) continues to32K. Do not treat16K
acceptance as32K evidence. The prior32K accepted speed remains9854.324387.
No global default promotion. Full raw service archive will be produced after
both lengths complete,without compressing alongside timed service waves.
