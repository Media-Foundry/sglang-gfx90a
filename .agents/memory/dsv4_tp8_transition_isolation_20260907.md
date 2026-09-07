# TP8 request-transition isolation after the BS1 speed checkpoint

Base97593289d6. Original weights, TP8/EP1/no-A2A, native AR,
M36864 throughput profile, graph tiers1/2/4/8/16/24/32, pool131072.
GEMV candidate remains opt-in. No performance win is claimed here.

## C1: scheduler overlap off -- no result

Changing only DISABLE_OVERLAP_SCHEDULE=1 fails the existing PrefillDelayer
startup assertion: the retained delayer requires overlap. The failed process
tree exited; GPUs were clear before restarting. Do not compare its speed.
Log: `/tmp/dsv4_tp8_woa_no_sched_C1_20260907.log`.

## C2: WAR gate actually on -- still stalls

Source audit found the scheduler's WAR gate defaults OFF on HIP. The previous
POST_REPLAY fallback fix therefore did not activate the barrier in A3/B3.
Preserve overlap/delayer and set SGLANG_ENABLE_WAR_BARRIER=1 only. Confirmed
the actual server environment contains WAR=1 and WOA=1.

New `check_dsv4_prefill_probe_transitions.py` reuses the six fixed-prefix
teacher probes from `/tmp/dsv4_ck_tp8_fixed2_c1.json`, sequential fresh salts,
temperature0, max_new_tokens1, input/output logprobs. It saves the pending
request before send, records responses after completion, and treats client
timeout as unknown server state (not proof that the server stopped).

C2 completed10 probes then stalled on round1/probe4, diverse-31, input45.
Repeated observation: GPUs0/1 at100%, GPUs2--7 idle; process remained alive.
The completed requests' next token IDs matched the reference. This does not
establish exact cached-decode logits. The short input prefill is below the
large CK selector; no large-prefill workload preceded this reproduction.
Although the probes request only the prefill token, possible overlap lookahead
must be checked before claiming absolutely no incidental decode was launched.

Logs/artifacts:
- `/tmp/dsv4_tp8_woa_war_enabled_C2_20260907.log`
- `/tmp/dsv4_tp8_war_C2_probes_20260907.json`

The scoped service tree was terminated after confirmed nonprogress; all GPU
processes cleared. POST_REPLAY plus enabling the WAR gate is not sufficient
to eliminate these stalls. Next C3 keeps that exact server configuration and
removes logprob requests only, using the same six prefixes and next-ID oracle.

## C3/C4: neither logprobs nor custom all-gather is required

C3 repeats the C2 service but the client omits logprobs. It stalled on the
first request (diverse-03/input46), with no completed response. Thus the
reproduction does not require input/output logprob processing.

C4 keeps the C3 workload/server settings and sets SGLANG_USE_AITER_AG=0,
retaining AIter all-reduce. Six probes completed with reference next IDs,
then round1/probe0 stalled. Repeated activity check showed GPUs2/4/5/6/7
busy,0/1/3 idle. Disabling custom AG alone is not sufficient either.
Both stalled process trees were stopped and GPU processes cleared before
the following launch. No timing is accepted from these diagnostic runs.

Artifacts: `/tmp/dsv4_tp8_war_C3_no_logprobs_20260907.json` and
`/tmp/dsv4_tp8_no_ag_C4_no_logprobs_20260907.json`.

C5 retains C4 settings and adds opt-in per-rank periodic Python stack dumps
to `/tmp/dsv4_tp8_stacks_C5_20260907`. This uses faulthandler, not ptrace or
GPU profiling. Diagnostics remain off by default and are not a performance
measurement configuration. Goal: identify each rank's host wait location
before further collective/scheduler changes.

## C5 diagnostic limitation and delayer decision bug

C5 reached ready, then exited before any probe was submitted. Logs contain
Gloo peer-closed exceptions in delayer negotiation; this is not evidence of
the original GPU stall's initiating cause. Partial stack dumps include idle
request-receiver/delayer collectives. Kernel logs were not readable. The
temporary periodic stack-dump code was removed, not committed or promoted.

Static inspection then found an independent, directly demonstrable bug:
the delayer gathers queue state but tests elapsed wall time locally afterward.
Ranks straddling the20ms deadline can disagree on whether to run prefill or
continue negotiating, issuing different collective sequences.

Fix: add one timeout bit to the existing gathered record (5 to6 int64 fields)
and use the shared TP0 timeout decision across DP groups. No extra collective.
Local elapsed time is retained only for diagnostic wait duration, not admission.

The real pre-fix method, extracted from HEAD via AST, returns `[False, True]`
for clocks10.004/10.006 with5ms deadline and identical gathered queue state.
The fixed method returns `[False, False]` when the shared timeout bit is0;
the regression fixture also requires `[True, True]` when it is1. This proves
the local-clock admission bug, but full E2E causality is still being tested.

D1 restores original B3 flags (AIter AG enabled, WAR gate default off), with
only the delayer consensus fix. No debug timers. Candidate GEMV stays ON for
testing, default OFF in code. Log:
`/tmp/dsv4_tp8_delayer_consensus_D1_20260907.log`.

D1 first result: all60 fixed-prefix probes completed. All60 next IDs, input
logprobs and output top-logprobs exactly match the reference. Earlier services
stalled after0/6/10 probes. Together with the old-method clock-skew reproduction,
this strongly supports admission disagreement as a cause; it is not a claim
that all possible communication races are resolved. Eight unit tests passed,
including the gathered six-field layout and TP0 selection contract. The test
class now uses unittest.TestCase to avoid unrelated datasets/benchmark imports.
Artifacts: `/tmp/dsv4_tp8_delayer_D1_probes_20260907.{json,log}`.
The full C1 speed/semantic workload and P32-to-C1 transition remain to be retested.
