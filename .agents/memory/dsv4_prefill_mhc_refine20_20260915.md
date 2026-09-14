# Comb-only continuation: exact component win, not production-wired

Following the completed config20 service ABBA (-1.1365% throughput), tested
the existing full20 FP16-Fn boundary against the existing full8 boundary plus
one small one-wave kernel performing12 additional row/column normalizations
of its FP32[4,4] comb. No repeated exp/initial normalization. No checkpoint,
pre-mix, post, or normalized-output precision change; no extra tensor workspace.

Only physical GPU4, after all three TP8 service trees stopped and amd-smi
reported no GPU owners. The oracle checks every returned tensor byte-exact
against full20 before accepting timing. It uses captured layer0 residual/Fn,
repeated rows, synthetic post/combine inputs; this is not a full-model oracle.

Initial screen,10 mutations/shape:

| M | full20 ms | full8+12 ms |
| ---: | ---: | ---: |
| 1 | .213656 | .236713 |
| 128 | .218249 | .236889 |
| 8192 | 3.238671 | 3.052376 |

Extended screen,25 mutations/shape, row permutation,1000 fixed-input graph
replays plus one changed-input replay per shape: all outputs bit-exact.

| M | full20 ms | full8+12 ms |
| ---: | ---: | ---: |
| 128 | .220344 | .243593 |
| 8192 | 3.264372 | 3.077998 |
| 32767 | 12.837846 | 12.078188 |
| 32768 | 12.938440 | 12.173517 |

All times are allocating eager **complete-boundary** event timings, three
ABBA cycles, five calls/event. Graph checks establish tested replay equality,
not graph throughput. The extra launch hurts small M; do not admit M1/M128
to a production optimization on the basis of this experiment.

Large-M boundary latency decreases about5.7–5.9% (roughly.76ms at32K).
This is versus full20, not versus the old faster8-iteration baseline. It
may recover most of the policy's extra cost, but E2E savings are unmeasured.
One read/write of16 FP32 values/token adds2MiB each direction atM32768.

Evidence: `.agents/experiments/dsv4_prefill_mhc_refine20_20260915/`
`candidate.py`, `oracle.py`, `screen.json`, `full.json`, README.
GPU handles4509 and35344 both exited0. Production integration and service
ABBA remain pending. If integrating, require active original-V4 TP8 ordinary
prefill config20 scope plus large-M guard; do not merely test global env20,
which could accidentally admit decode or another worker. Preserve small-M
full20 and all unscoped behavior. Keep experimental default-off until verified.

## Follow-up: stronger inputs and production selector

`parameter-mutations.json`: M1/128/8192,100 mutations each of local fixture
Fn/scales/base/residual (not checkpoint files), plus1000 graph replays and
changed-input replay per shape, all byte-exact. No model weights were edited.

Default-off `SGLANG_DSV4_PREFILL_MHC_COMB_REFINE20=1` is now wired only inside
the existing config20 context and only8192<=M<=65536. AR/draft/TP4/V4.1 and
small-M remain outside; all small-M scoped prefill still computes full20.
40 CPU scope/launch tests pass. `integrated.json` checks actual refinement
call counts (not just equal outputs that might hide fallback), all outputs,
row permutation and1000 graph+mutation replays for M1/128/8192/32767/32768/65536.
All pass; integrated full20->refined times at M32768 are12.940247->12.175606ms,
and M65536 25.369091->23.848753ms. GPU handle82494 exited0.

Independent service ABBA is prepared in
`.agents/experiments/dsv4_prefill_mhc_refine20_service_20260915/`.
All arms hold config20,wide-query16,1M KV,32K chunk and original weights fixed;
only the refinement flag differs. It is not the earlier8-vs20 policy trial.
No service throughput result for the refinement is established yet.
