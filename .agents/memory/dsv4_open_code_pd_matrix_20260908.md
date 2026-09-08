# Open-source code P/D matrix (in progress)

User replaced Mooncake anonymized trace replay with real open-source code P/D.
Required matrix: TP4 and TP8; C1/2/4/8/16/32; three measured rounds each;
decode AR and DSpark separately. Untuned decode tiers may receive at most
2–3 bounded optimization attempts, with correctness after every modification.
Original checkpoint and KV capacity remain priorities. This is a new benchmark
task, not a resumption of the stopped A2 optimization sweep.

## Frozen workload

Builder: scripts/rocm/build_dsv4_open_code_pd.py, git-show source from public
Media-Foundry/sglang-gfx90a revision 54b93c45c2ad7eb2d6583c896d98b4d7fcee3d20.
No uncommitted source is used as workload content. Official checkpoint chat
encoder; 32 distinct source-review requests, including source paths, hashes,
public URLs, complete prompt text and IDs. Source candidate lists include
files concatenated before token-length trimming; not every candidate is
necessarily present in the final excerpt. License remains the source repo's.

Generated with DS Python, --output-dir /tmp/dsv4_open_code_pd_20260908:
- decode.json: 511–512 input tokens; SHA256
  12494eea5a4b77379561e4b5f34e2b4f869d4b7563ad25e047aed2d1bf89dd44
- prefill.json: 8191–8192 input tokens; SHA256
  3ff1c2bdc5c286d15b4c197984073dce86f76cdddad54d2fb4866f2d6a69e85f

Input integrity/uniqueness and Python compile checks passed; prefill CLI now
accepts C2 as required. No model kernel or serving configuration changed.

## Important rejected measurement

Initial controller /tmp/dsv4_open_code_pd_tp8_ar_20260908 used the older
ignore_eos=True harness for 2048 outputs. The C1 answer naturally stopped,
then forced continuation emitted repeated EOS. That throughput is excluded
from the requested real-code results. Controller PID3679355 was interrupted;
service PID3656477 was preserved. Do not combine those artifacts with the
natural-output matrix.

## Current natural-output protocol

scripts/rocm/bench_dsv4_open_code_decode.py respects EOS, caps output at2048,
and records IDs, readable output, finish reason and acceptance metadata.
Each synchronized wave's common resident window starts at latest first token
and ends at earliest last token. Thus prefill and batch drain are excluded;
this is streaming wall-time throughput, not pure GPU kernel time. Accumulate
at least30 seconds of common windows per measured round, three rounds; an
extra warmup wave is excluded. Requests rotate deterministically through the
fixed manifest. The number of waves is duration-dependent: cross-config final
analysis must disclose request-mix counts and use matched waves if different
durations result in materially different request mixes. Do not silently call
different case distributions a controlled acceptance comparison.

Prefill is a separate invocation with max_new_tokens=1 and fresh cache salts.
Input tok/s is total input divided by wave start to last first token, including
admission and first-token overhead. This is not GPU-only prefill time.
8K is an initial length screen, not proof of peak throughput; increase lengths
if needed after inspecting the curve and memory budget.

Current controller: scripts/rocm/run_dsv4_open_code_pd_arm.py, session77409.
Output directory: /tmp/dsv4_open_code_pd_tp8_ar_natural_20260908.
TP8 AR service PID3656477, loopback30011; 1M pool, mem0.96, cap16,
chunk/max-prefill36864, attention overlap/gate prefetch/down-uniform/fixed
warmup/AR blocks4 retained. AMD PID ownership checks before each subprocess.
Warmup C1 produced1233 tokens and a natural EOS with readable technical prose;
the excerpt is limited, and its proposed issue is a hypothesis, not validated
repository correctness. Warmup speed is not a measured-round result.

Remaining: finish/audit TP8 AR, establish sufficiently long matched workloads,
run TP4 AR and TP4/TP8 DSpark, three rounds per group; inspect actual graph tiers,
capacity and semantic witnesses. Historical TP4 DSpark speed profile contains
anchor-only routed/bonus-row approximations: label and review explicitly,
never present it as strict target verification merely because weights are
unchanged. No new DSpark or TP4 service has been started at this checkpoint.
