# TP8 native decode Top-K geometry screen

Parent: 488b148e6d. No production changes or default changes.

Current DS AIter has no topk_gating export. topk.py catches ImportError and
uses moe_fused_gate; its native grouped router switch remains wired inside that
module (it is not a disconnected switch). Current service sets it to zero.

Resource check via amd-smi: all listed GPU PIDs belonged to retained service
2727188 or its children; no external GPU job. Isolated tests used physical GPU4,
with no simultaneous HTTP benchmark. No model weights were cached or expanded.

Reused existing bench_dsv4_m64_topk_geometry.py via a small row-count adapter
bench_dsv4_decode_topk_geometry.py, overriding M to1/32. Random BF16 scores and
random BF16 bias, N256/K6/sqrtsoftplus, renormalize, scale1.5. This is a synthetic
screen, not a real-checkpoint correctness oracle. Each candidate: 100 mutations
including equal/adjacent inputs, 1000 graph replays, five ABBA cycles, 500replays
per sample and30warmups. Table is trimmed mean in microseconds.

| Rows | Candidate | Baseline | Candidate | Exact weights /100 |
|---:|---|---:|---:|---:|
|32|BM1/W2|7.2317|10.1452|0|
|32|BM1/W4|7.2502|10.2102|0|
|32|BM2/W1|7.2299|8.4772|1|
|32|BM4/W1|7.2418|11.1159|1|
|1|BM1/W2|7.1553|9.9541|79|
|1|BM1/W4|7.1788|10.0904|78|
|1|BM1/W1/WPE2|7.1505|7.0973|100|

All candidate IDs exact100/100 and all stable on1000replays. The C1 WPE2 saving
is only0.053us, insufficient evidence for service promotion. No candidate was
enabled and no E2E gain claimed. Current kernel's ~7us standalone cost must not
be confused with ~13–15us marker ranges containing dependencies/queue time.
This confirms the older M64 rejection for the newly tested M1/M32 row counts.

Raw sample arrays:
/tmp/dsv4_tp8_m32_topk_screen_20260908.jsonl
/tmp/dsv4_tp8_m1_topk_screen_20260908.jsonl

Retained service has C1 prep fusion off, accepted M32 attention overlap on.
Withdrawal correctness previously passed all12 fresh-prefix probes (IDs, input
logprobs, output top20logprobs). No production source changed after that check.
