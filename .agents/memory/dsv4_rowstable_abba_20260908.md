# Row-stable prefill service ABBA (complete)

Controller: `scripts/rocm/run_dsv4_rowstable_abba.py`.
Live state/result paths: `/tmp/dsv4_tp8_rowstable_abba_20260908.json`.
A is the row-stable prefill candidate; B is flag-off baseline. Order A,B,B,A.
Native TP8/EP1/no-A2A, original weights,1048576 KV pool, same graph tiers.
Only ROW_STABLE_PREFILL changes. No debug dump flags or external GPU jobs.

Each block runs C1 with one warm pass plus two measured passes on three cases,
then six C32 waves of the fixed32 diverse code prompts,256 tokens/request.
Score C1 measured reps only and C32 rounds1..5, discarding first wave for every
block whether or not cold work is visible. Retain cold observations separately.
C32 full output IDs are saved for reference and cross-round checks.
Compare candidate endpoint blocks against the two middle baseline blocks;
do not treat earlier noninterleaved numbers as ABBA evidence.

Controller initially used the shell HTTP proxy for health checks and had not
started a measurement. It was stopped after verifying zero blocks/children;
the model service was retained. Health checks now bypass proxies like the
benchmark scripts, and the fresh controller started measurements normally.

No acceptance or speedup conclusion yet. Numerical fixture improvements and
remaining decode batch-slot differences are documented in adjacent records.

## Completed result

| Block | Flag | C1 case-geomean | C32 warm E2E | C32 warm resident |
|---|---:|---:|---:|---:|
| A0 |1|84.2815|985.6892|1031.3966|
| B1 |0|84.3796|976.8440|1023.9750|
| B2 |0|84.1978|977.6094|1024.2727|
| A3 |1|84.7652|981.5878|1027.6005|

Geometric means of the two blocks per arm:
C1 candidate84.5230 vs baseline84.2887 (+0.278%);
C32 E2E983.6364 vs977.2266 (+0.656%);
resident1029.4968 vs1024.1239 (+0.525%).
The effect is small and endpoint variation is visible; do not claim a major
optimization or statistical confidence from this single ABBA cycle.

All24 measured C1 full completions equal the historical reference.
All768 C32 requests have256 IDs, finish length, and saved IDs reproduce their
SHA256. Cross-round exact full C32 requests are8/32 and3/32 in the candidate
blocks versus0/32 in both baseline blocks. Counts alone are not semantic
correctness, and the candidate is not fully bitwise stable in decode.

Structured results are in `dsv4_rowstable_abba_20260908_summary.json`; raw logs,
full IDs and block paths remain in the controller state under /tmp.
Current service PID3013729 is the final candidate arm with1M pool, dumps off.
Repository default remains off; this screen supports no material speed penalty
for the fixed-prefill numerical improvement, not completion of the TP8 goal.
