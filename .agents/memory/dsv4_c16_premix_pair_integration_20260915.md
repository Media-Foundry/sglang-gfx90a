# Paired-column FP32 pre-mix: exact integration and accepted service improvement

Independent-column candidate source and component evidence were committed at
4ee7ff12ab. M32768 pre-mix 7.63918 -> 4.98918 ms, approximately 34.7% less time,
not an E2E result. Per-row K1024 reduction and FP32 arithmetic are retained.

Production integration uses a new default-off MIX_PAIR_COLUMNS flag. It additionally
requires the existing mix reuse scope, original V4, TP8/EP1/no CP/PP/DCP, ordinary
EXTEND, non-draft/no speculative/TBO, no capture, group8 and M8192..65536.
The older reuse predicate is not tightened as part of this experiment. No new
tensor workspace. Existing batch-one FP16 Fn fused-tail priority remains unchanged.

Six CPU tests passed. Isolated physical GPU4 production-wrapper oracle completed
(integrated.json): M1/128/8191/8192/32767/32768/65536/65537, group4 and8,
exact outputs for flag off/on. Counted real kernel launches prove positive and
negative dispatch; large-M actual MHC caller also passed ten random x/Fn mutations
per shape, all FP32 output bits equal, scope contexts restored. These are not
fresh full-model activation checks or a claim that global service drift is fixed.

Fresh A1/B1/B2/A2 C16x8K service trial completed (sweep session35562 exit0), three
formal waves per leg plus separate answer review. A1/B/A2 PIDs1543098/1550995/
1558118 all stopped with remaining=[]; all eight GPUs were confirmed free.

Extended CPU regression suite:45 passed, with one unrelated pytest asyncio_mode
warning. The initial run found the old AST-execution test fixture omitted the new
module-global pair predicate; adding its default None fixed the fixture without
altering production code. Three separate client tests pass, checking first-token
versus drain timing and rejecting wrong input echoes or empty output IDs.

## Accepted service result and narrowly scoped promotion

A1/B1/B2/A2 medians:6959.4535 /7368.9612 /7366.3857 /6964.3819 input tok/s.
Mean leg medians **6961.9177 -> 7367.6734 (+5.8282%)**. This is total131069
input tokens divided by the wave's last first-token time minus earliest start,
not resident decode or drain throughput. All192 formal input echoes and96 quality
echoes match; zero prefix hits. Same original V4 weights, TP8/EP1, 1M logical KV,
32K chunk, query16/runtime-M, mix8. Config20/refinement/wide-C4 stayed off.
The recorded launch scripts differ only in MIX_PAIR_COLUMNS. Eight B ranks log
the actual candidate at M32767; A ranks do not. Formal log intervals contain no
compile or exception lines and each has12 four-request/page-rounded32K admissions.
The latter is not a proof of exact M for every forward.

Quality: A1 repeats15/16, B16/16, A216/16; every wave has16/16 matching first
tokens. Both candidate waves match A1 quality0 and both A2 waves completely.
All candidate texts and unique control alternatives were read: coherent source
analyses, no observed new collapse within128-token excerpts. Generated claims
about source bugs were not generally fact-checked. A1's case8 wording alternate
persists despite identical input IDs, proving the baseline is not globally
deterministic; neither the component exactness nor matching answers fixes this.

Archive:106 files,2460276 bytes, SHA256
`3ebf75a6b3d46ff16f9760deae965a235ae424abecfa43ab469e699de95376bd`.
Directory:`.agents/experiments/dsv4_c16_premix_pair_service_20260915/`.
The archive retains tested production sources before launcher promotion.

Only the combined TP8 multi-request + prefill-throughput profile now defaults
MIX_PAIR_COLUMNS=1. Explicit0 wins. Runtime retains original-V4/ordinary-EXTEND/
no-spec/no-draft/no-CP/PP/DCP/TBO/no-capture, group8 and M8192..65536 guards.
The standalone wrapper remains default-off. No weight precision or workspace
change. Do not reuse the old2.70s pre-mix profile as remaining budget: it predates
this improvement. Next update the full service profile before choosing another
large kernel project; whole-model drift remains separately unresolved.

Post-promotion validation:46 CPU scope/default tests and3 client tests passed;
bash syntax and diff checks passed. The same unrelated asyncio_mode warning
remains. Promotion changes only an environment default, not tested arithmetic.
