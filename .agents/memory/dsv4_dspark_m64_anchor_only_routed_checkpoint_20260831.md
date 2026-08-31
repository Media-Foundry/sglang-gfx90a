# DSV4 DSpark M64 anchor-only routed-MoE checkpoint

Date: 2026-08-31

## Idea and scope

For gamma-one target verification, M64 is ordered as
`[anchor_0, draft_0, ..., anchor_31, draft_31]`.  The anchor row determines
whether the current draft token is accepted; the draft row only supplies the
bonus token when that draft is accepted.  The accepted approximation keeps
the complete target model on every anchor row, but removes routed-expert work
from draft rows.  Draft rows still execute the shared expert and all other
model components.

The selector requires every condition below:

```text
gfx90a
ForwardMode.TARGET_VERIFY
ForwardBatch.batch_size == 32
spec_info.num_tokens_per_req == 2
hidden_states.shape == [64,4096]
explicit environment opt-in
```

Consequently native AR BS64, other speculative widths, other batch tiers and
other architectures cannot enter it.  Before the grouped FP4 sorter, odd-row
top-k IDs are replaced with the already-supported `-1` sentinel.  Immediately
after routed experts, odd routed outputs are zeroed before shared-expert add
and TP all-reduce.  Original checkpoint weights are unchanged.

This is not bitwise target inference: bonus logits are approximate.  It is a
DSpark speed/quality tradeoff and must never be reported as native AR.

## Verification

The pure selector unit test covers positive reachability and negative guards
for AR mode, BS16, speculative width four, M32 and missing spec metadata.
It passed.  All tested DSpark services passed the official France first-nine
token oracle and semantic Paris check.  All 32 distinct concrete coding
requests in every long round generated exactly 256 tokens with
`finish=length`.

A separate native-AR negative-control service was then started with the
approximation environment variable deliberately set to one.  Since its
`ForwardMode` is not `TARGET_VERIFY`, the branch remained unreachable.  One
round of 32 distinct requests generated 64/64 tokens for every request,
finished with `length`, and passed France exact/Paris; `spec_accept_length` was
absent as expected for native AR.  Its resident rate was 712.88 tok/s, recorded
only as reachability evidence rather than a performance comparison.

Physical GCD 4 was occupied by an unrelated BIO training process, so the
directional B-A-B used the otherwise idle physical GCDs 0--3.  Arms are
compared only within that matched card group; they are not mixed with prior
4--7 results.

## B-A-B results

Three rounds per independent service, 32 distinct coding prompts and 256
generated tokens:

| arm | resident tok/s | accept length | host step ms | scheduler tok/s |
|---|---|---|---|---|
| B1 anchor-only | 777.404, 778.170, 839.990 | 1.5334, 1.5643, 1.6554 | 56.367, 58.576, 53.554 | 610.425, 623.318, 614.404 |
| A full routed | 722.416, 715.648, 718.771 | 1.6290, 1.6111, 1.6147 | n/a, 61.996, 63.204 | n/a, 553.571, 578.453 |
| B2 anchor-only | 871.818, 825.866, 820.471 | 1.6877, 1.6130, 1.5990 | n/a, 55.058, 56.977 | n/a, 623.860, 641.117 |

Service resident medians:

```text
B1 = 778.170 tok/s
A  = 718.771 tok/s
B2 = 825.866 tok/s
median(B1,B2) = 802.018 tok/s
gain vs A = +11.582%
```

The speedup comes from lowering the target step by roughly 7--9 ms, while the
approximate bonus path usually reduces accepted length by about 3--5%.  The
net resident throughput remains positive in both independent candidate
services.

## Decision and limitations

Enable by default only for `start-dspark` under the TP4 BS32 profile.  The
environment variable remains an explicit rollback (`=0`) to recover exact
target bonus logits.  Keep it out of native AR and out of any correctness
claim stronger than the France/code completion gates above.

This checkpoint raises the matched-card center only to about 802 tok/s, so it
does not satisfy the 1.5k goal.  Continue the exact A4 CTA weight-multicast
oracle in parallel; its savings may stack on the remaining anchor routed work.

Artifacts:

```text
/tmp/dsv4_anchor_only_b_smoke.json
/tmp/dsv4_anchor_only_b1_code32.json
/tmp/dsv4_anchor_only_a_code32.json
/tmp/dsv4_anchor_only_b2_code32.json
/tmp/dsv4_anchor_only_ar_negative_code32.json
/tmp/dsv4_anchor_only_{b,a,b2}.log
/tmp/dsv4_anchor_only_ar_negative.log
```
