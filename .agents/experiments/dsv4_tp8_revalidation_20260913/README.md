# Original DeepSeek V4 Flash: TP8 native-AR P/D revalidation

The user cancelled DSpark measurements during bring-up. The requested matrix
is **TP8, C1/2/4/8/16/32/64, original checkpoint, native AR only**.
V4.1 remains frozen. Historical strict-DSpark1127 and approximate TP4-DSpark1500
are not reference AR measurements.

## Artifacts

- `start-ar-matrix.sh`: exact launch,1M pool, seven graph tiers, no speculation.
- `manifests64/`: fixed public-source requests, complete text and input IDs.
  First32 cases are identical to the historical32-case corpus.
- `legacy-manifests/decode.json`: historical corpus used by the initial pilot.
- `ar-france.json`: semantic sentinel response and completion IDs.
- `ar-pilot-c*-measured.json`: two short natural-output regression waves.
- `ar-matrix/state.json`: live authoritative progress; missing cells are pending.
- `ar-matrix/{prefill,decode}_c*_measured.json`: three-round measurements.
  Corresponding warmup files are excluded, not silently merged into results.
- `ar-summary.json`: recomputed checked summary, generated with the command below.
- `ar-final-decode/state.json`: final seven-tier native D rerun after the exact
  empty-indexer-tile fix. Three measured rounds/tier; separate warmups.
- `empty-tiles-A1/B1/B2/A2.json`: independent long8K C32 service ABBA;
  resident169.657/611.103/610.853/169.843tok/s. Not the short512-input matrix.
- `empty-tiles-integrated.json`: production-wrapper score-bit and logical/
  physical Top-K oracle, seven cases,100mutations/case and checks after every
  one of1000graph replays/case. Components are not whole-model equivalence.
- `empty-tiles-Final-service.json`: new process launched without an explicit
  empty-tile flag; the validated TP8/EP1/no-A2A launcher supplies its default1.
- `final-vram-snapshots.jsonl`: separate5s observations for that final process.
- `build-final-report.py`: after final D completes, rechecks all21 P and21 D
  rounds from raw evidence, produces `RESULTS.md` and `final-report.json`.
  P remains the original completed matrix because the new guard excludes
  prefill entirely. Baseline and final D are both retained.
- `ar-matrix.service.log`: startup, pool capacity, actual kernel/graph selections.
- `build-witnesses.jsonl`: compiler children observed after the watcher started;
  it cannot establish absence of in-process compilation or earlier builds.
- `ar-vram-snapshots.jsonl`: appended memory observations during decode.
  The earlier `ar-vram-samples.json` contains cumulative CLI history,
  159 observations/GPU from23:48:31 through00:01:46, starting during P16.
  An earlier claim that it held only the last snapshot was wrong and has
  been corrected. Neither file establishes an exact allocator peak.
- `audit-vram.py` / `vram-summary.json`: independently summarize both sources.
- `audit-window-context.py` / `context-window-diagnostic.json`: post-hoc
  first-wave timing before generated1400 (input<=512). Diagnostic only; never
  substituted for the formal three-round natural-output median.
- `check-post-matrix.py`: refuses while matrix runs; afterwards sends France
  and three8K source cases twice, saving readable output for bounded review.
- `validate-output-text.py`: CPU-only independent tokenizer check of saved
  completion IDs against response text, for every completed warmup/measured
  cell; saves per-file hashes in `output-text-validation.json`.
- `pack-results.py`: after the owned service stops, archives JSON/log evidence
  and checks every archived file against its SHA256. The resulting
  `measurement-evidence.tar.gz` and `evidence-index.json` preserve raw data
  without committing hundreds of MB of repetitive streaming JSON to Git.

If using a fresh checkout with archived evidence, extract in this directory:

```bash
tar -xzf measurement-evidence.tar.gz
```

The archive is generated only after the matrix is complete and the test
service has stopped; an in-progress checkout may not contain it yet.

## Measurement definitions

P:8K source inputs, fresh cache salts, one generated token. Input throughput
is total prompt tokens divided by wave start to last first-token time. C is
client concurrency; admission is capped16 and chunk rows36864, not C-sized
GPU prefill. The accepted BF16-CK prefill profile changes execution arithmetic
but does not modify checkpoint files; it is not SDOT bitwise equivalence.
This8K/1M-pool table is not the earlier2304-token/131072-pool6.42k profile.

D:512-token inputs, temperature0, natural EOS, at most2048 output tokens.
Each wave's resident interval starts at the latest first token and ends at the
earliest last token. Each round accumulates at least30s of these intervals.
Separate HTTP wall throughput includes admission and drain. Case selection
rotates deterministically through64 cases; natural answer length can change
the number of waves and their context mix. Record case counts rather than
assuming identical distributions across concurrency tiers.
Fully resident requested tiers have matching captured rows. During drain,
intermediate sizes round up (e.g.33--63 requests use M64); this is counted in
whole-wave HTTP throughput, not represented as a fully resident C64 sample.

All requests retain readable text, completion IDs/hashes, finish reasons and
raw streaming timestamps/counts. These integrity checks and France are bounded
correctness evidence, not a whole-model bitwise or factual code-review proof.
The source excerpts are deliberately length-limited and sometimes truncated.

## Recompute the table (does not contact the service)

From the repo root:

```bash
/home/pc/anaconda3/envs/DS/bin/python scripts/rocm/summarize_dsv4_open_code_matrix.py \
  .agents/experiments/dsv4_tp8_revalidation_20260913/ar-matrix \
  --output .agents/experiments/dsv4_tp8_revalidation_20260913/ar-summary.json
```

Full launch/progress/regression history is in
`../../memory/dsv4_tp8_revalidation_20260913.md`.
