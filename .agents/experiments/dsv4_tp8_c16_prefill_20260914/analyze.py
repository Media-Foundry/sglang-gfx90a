"""Summarize completed arms; never substitute historical results for a control."""
import hashlib
import json
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parent


def main():
    arms = {}
    legs = {}
    for arm in ('A1', 'B', 'A2'):
        root = ROOT / arm
        done = json.loads((root / 'complete.json').read_text())
        plan = json.loads((root / 'plan.json').read_text())
        validation = json.loads((root / 'validation.json').read_text())
        log = (root / f'P16-{arm}.service.log').read_bytes()
        arms[arm] = dict(plan=plan, validation=validation)
        for record in done['records']:
            if not record['measured']:
                continue
            rows = json.loads((root / (record['name'] + '.json')).read_text())['rounds']
            section = log[record['log_start']:record['log_end']].decode()
            batches = [line for line in section.splitlines() if 'TP0] Prefill batch,' in line]
            shapes = [int(re.search(r'#new-token: (\d+)', line)[1]) for line in batches]
            rates = record['rates']
            legs[record['name']] = dict(
                rates=rates, median=statistics.median(rates),
                range_pct=100*(max(rates)-min(rates))/statistics.median(rates),
                wall_s=[row['prefill_wall_s'] for row in rows],
                total_input_tokens=[row['total_prompt_tokens'] for row in rows],
                prefill_shapes_in_log_order=shapes, prefill_log_lines=batches,
                completion_ids=[row['completion_ids'] for row in rows],
            )
    assert len({a['plan']['manifest_sha256'] for a in arms.values()}) == 1
    control = statistics.mean(legs[x]['median'] for x in ('A1', 'A2'))
    candidate = statistics.mean(legs[x]['median'] for x in ('B1', 'B2'))
    summary = dict(
        metric='TP8 C16 input tokens / first request start to latest first token; 1 output token',
        arms=arms, legs=legs, control_mean_leg_median=control,
        candidate_mean_leg_median=candidate, gain_pct=100*(candidate/control-1),
        control_return_pct=100*(legs['A2']['median']/legs['A1']['median']-1),
        warning='B1/B2 share one service; this is a bounded screen, not independent process replication of B.',
    )
    artifacts = {}
    for arm in arms:
        for path in sorted((ROOT/arm).iterdir()):
            if path.is_file():
                artifacts[str(path.relative_to(ROOT))] = dict(
                    bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    summary['artifact_manifest'] = artifacts
    (ROOT/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({key: value for key, value in summary.items()
                      if key not in ('arms', 'legs', 'artifact_manifest')}, indent=2))


if __name__ == '__main__':
    main()
