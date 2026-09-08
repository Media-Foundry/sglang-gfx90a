# Homogeneous-prefix control: stable multiset, client-slot dependence

Same unchanged1M TP8 server2857900. Three barrier-start waves of32 requests,
all the same46-token prefix (established teacher-forced probe0), fresh salts,
one output token plus logprobs. Not throughput and not cached-decode testing.
Server reports two16-request prefill groups per wave. Reported padded token
count4096 is not proof of actual kernel M;46*16=736 logical input rows.

96/96 next IDs match sequential reference. As in heterogeneous testing,
per-client-index logprob arrays are not cross-round bitwise exact. However,
each wave contains exactly16 distinct input-logprob arrays and16 distinct
output-top20 arrays, each appearing twice. Their complete multisets are
identical across all three waves, including multiplicities.

Therefore this observation supports deterministic batch-position dependence
plus reassignment of clients to scheduler slots, rather than unstructured
run-to-run numerical noise. It does not identify the responsible operator or
exclude every race. The sequence of operations is not fully fixed merely by
holding the client ID fixed. One possible next diagnostic is a layer-local
repeated-row input fixture, checking raw projection/MoE/attention rows before
attributing differences to atomic reduction or recently accepted decode flags.

Top1 logprob range across slots:-1.072883e-6 to-4.768371e-7; sequential value
-5.960463e-7, same next-token2280. This highly confident token is a weak semantic
oracle; prior low-probability top20 differences cannot be dismissed from this.

Added per-probe multiset comparison to the diagnostic harness so future runs
do not confuse request identity with physical batch position. No production
model/kernel edits. Raw artifact:
/tmp/dsv4_tp8_1m_homogeneous_probe0_20260908.json
Client97736 completed. Existing artifact predates automatic multiset summary;
the multiset result above was computed directly from its full response arrays.
