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
