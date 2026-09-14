# Small TP8 native-AR down-consumer trial

User request: first try C32; only if beneficial, try C1 for a second point.
Original DeepSeek V4 Flash checkpoint, eight GCDs, TP8/EP1/no-A2A, 1M allocated
logical KV pool. DSpark is disabled. This is not an M128 padding experiment.

The new default-off `SGLANG_DSV4_GFX90A_TP8_M32_DOWN_CONSUMER` uses the existing
CTA16 consumer and fixed Top6 reduction. Its native-decode, TP8/EP1, M32/I256,
A4/R2/W8/B832/LDS guard is independent of the older shape-only flags. It
replaces only the down quantization/compute chain. Current gate-prefetch,
attention overlap, collectives, C1 wo_a GEMV, original weights and prefill
settings are unchanged. A is the accepted uniform-metadata down path.

## Component

`component.py` runs on physical GPU4, refuses external owners, varies BF16
intermediates, packed weights, nonconstant physical E8M0 scales, routing and
route weights. Diverse and skewed regimes each have100 mutations and1000
checked graph replays. Both FP32 partial and BF16 output are checked with
`torch.equal` (numeric element equality, not a signed-zero-bit test).
This is synthetic and does not prove complete service numerical equivalence.
It times quant+down+reduction, not the full gate/routed stage or whole model.

## Service ABBA

`trial.py --arm A1`, then `--arm B`, then `--arm A2`, each in the DS conda env.
Each command creates a new, PID/birth/command-owned tmux service, checks one
readiness API after the log says ready, performs France and an excluded C32
warmup, measures its legs, and shuts down that exact service in `finally`.
B measures B1 and B2 within the same resident service. Each measured leg uses
two waves of the SAME32 public-source512-token inputs, max2048 output,
temperature0, natural EOS, unique salts. Resident throughput excludes
admission and drain; full HTTP throughput is saved separately. Seven native
graph tiers1/2/4/8/16/32/64 remain captured in all arms. Runtime logs must
witness raw_bs32/executed_rows32/input_rows32 and candidate selection in B.

This is a bounded ABBA screen, not the previous formal three-round matrix.
`analyze.py` independently decodes every saved completion ID, recomputes rates
and preserves each wave's duration and inputs hash. Semantic review is still
needed; differing hashes alone cannot diagnose the known batch/admission drift.

## Tests and limitations recorded before results

Two new CPU-only scope/selector tests and two existing DSpark isolation tests
pass. Running the entire older `test_dsv4_dspark_ar_guard.py` additionally
finds a pre-existing stale test: `test_dspark_ar_narrow_graph_guard` omits the
now-required `configured_blocks` argument. Neither that function nor that test
was changed by this trial. Do not report the entire old test file as passing.

The C1 direct kernel ALREADY quantizes BF16 intermediate into CTA-local LDS
before its SDOT down calculation; unlike grouped M32, it has no separate
second quantization launch to remove. It also uses nearbyint/clamp semantics
different from grouped quantization. A C1 follow-up must not simply force the
M32 consumer or claim its M32 saving transfers unchanged.

If the completed C32 ABBA is positive, `--arm A2 --follow-c1-if-beneficial`
reuses that return-control service for a separately recorded C1 A1 point.
Then `--arm B --c1-regression` and `--arm A2 --c1-regression` finish C1 ABBA.
This is explicitly a **C1 scope-isolation/regression point**, not an adapted
M1 candidate or a claimed C1 speedup: its native M1 graph cannot hit M32.
Run `analyze.py --concurrency 1` to analyze it separately. Its two waves repeat
the first public-source case, whereas C32 uses the complete fixed32 corpus.
