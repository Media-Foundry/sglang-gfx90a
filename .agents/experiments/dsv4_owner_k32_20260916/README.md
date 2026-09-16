# Exact K32 owner logits: measured C16x32K +2.62%

Original DeepSeek-V4-Flash, TP8/EP1/no-A2A, native AR, original checkpoint,
1,048,576 logical KV tokens, 32,768 prefill budget. Sixteen heterogeneous
public-code requests contain 524,286 actual input tokens per wave, zero cache
hits. Throughput is total input tokens divided by earliest request start to
latest first-token arrival, not decode or accepted-token throughput.

| ABBA leg | Median input tok/s (3 waves) |
|---|---:|
| A1, current owner K16 | 9602.610483 |
| B1, owner K32 | 9856.007239 |
| B2, owner K32 | 9852.641535 |
| A2, current owner K16 | 9603.059131 |

Control center9602.834807; candidate9854.324387; gain2.618910%. A1→A2 drift
0.004672%. Each leg range is below0.06%. B1/B2 share one candidate process;
controls start in independent processes. Each process runs four128-output-token
quality waves plus fixed continuation checks.192 answers are identical;
1008 teacher positions have exact logprobs and Top5 records, including versus
the preceding accepted common-FP32/20 checkpoint.

K32 uses the same query16 group, BF16 dot and MFMA16 instruction shape, with
32-key rather than16-key tiles. Compressor/cache updates, full query producer,
TopK tie rules, integer index exchange and physical-page mapping are unchanged.
Only admitted original-V4 ordinary prefill owner calls use it. Supported cache
preshuffle is0/16; FP16dot,FNUZ and other layouts fall back. Native decode,
draft/verify and V4.1 have no new dispatch path. No global default promotion.

The first diagnostic at395b95dd18 did NOT hit K32: the service used preshuffle16
while the prototype only accepted0. Its failed assertion and raw evidence remain
in `failed-check-evidence.tar.gz`. Separate layout16 stress/eight-rank checks
preceded the corrected `check-v2`, which actually selected K32 on all ranks and
passed2688 score/logical-ID/physical-ID comparisons. Neither diagnostic's tok/s
is a performance result because it runs full reference calculations too.

`validated-launcher.sh` is byte-identical to measured B. Launch with `bash`
only after ensuring no other service occupies these GPUs/port. It retains this
machine's source-hashed CK/IPC modules and explicit 1M-pool configuration.
Raw ABBA evidence is archived by `archive.py`; source hashes, startup, client
timestamps, token IDs, teacher records and owned cleanup are included.

8K/16K K32 regressions are complete in the separate regression directory:
8K10217.40→10282.76(+0.64%),16K10018.89→10151.65(+1.33%). Each length
passes192 identical128-token answers and1008 exact teacher positions against
its controls and preceding checkpoint. Three scored waves/ABBA leg, all8 ranks
hit the candidate,1Mpool and zero cache retained. See that directory's separate
summaries/acceptance/archive;32K acceptance.json remains its original snapshot.
The old8K10.206k versus old32K9.605k difference is input-length cost, not a
same-length regression; this result improves32K only. No universal precision,
quality or arbitrary-batch determinism claim follows from this finite fixture.
