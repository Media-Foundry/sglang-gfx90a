"""Summarize both AMD CLI cumulative history and append-only snapshots."""
import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo


def local_time(value):
    return datetime.datetime.fromtimestamp(value, ZoneInfo('Asia/Hong_Kong')).isoformat()


def main():
    root = Path(__file__).resolve().parent
    old = json.loads((root / 'ar-vram-samples.json').read_text())
    assert isinstance(old, list), 'CLI file is cumulative history, NOT one snapshot'
    observations = [('cli_history', row['timestamp'], row) for row in old]
    for line in (root / 'ar-vram-snapshots.jsonl').read_text().splitlines():
        row = json.loads(line)
        gpus = row['gpus']
        if isinstance(gpus, dict):
            gpus = gpus['gpu_data']
        observations.extend(('jsonl', row['time'], gpu) for gpu in gpus)
    groups = []
    for source in ('cli_history', 'jsonl', 'combined'):
        for gpu in range(8):
            items = [(t, g['mem_usage']) for s, t, g in observations
                     if g['gpu'] == gpu and (source == 'combined' or s == source)]
            assert items
            peak_t, peak = max(items, key=lambda pair: pair[1]['used_vram']['value'])
            assert peak['used_vram']['unit'] == peak['total_vram']['unit']
            groups.append(dict(source=source, gpu=gpu, samples=len(items),
                               first=local_time(min(t for t, _ in items)),
                               last=local_time(max(t for t, _ in items)),
                               peak_time=local_time(peak_t),
                               used=peak['used_vram'], total=peak['total_vram'],
                               percent=100*peak['used_vram']['value']/peak['total_vram']['value']))
    result = dict(caveat='Observed ~5s samples, not exact allocator peak. CLI history begins during P16; earlier P groups are not covered.',
                  groups=groups)
    (root / 'vram-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    for row in groups:
        print(row['source'], row['gpu'], row['samples'], row['used'],
              round(row['percent'], 3), row['peak_time'])


if __name__ == '__main__':
    main()
