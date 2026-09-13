"""Post-hoc context diagnostic, NOT a replacement for the formal D metric.

Inspect the first measured wave at each completed concurrency. Stop before
generated token 1400 (input <=512), keeping all requests below raw2048.
Natural EOS still bounds the common window. No new service traffic is sent.
"""
import bisect
import json
from pathlib import Path


def count_at(samples, timestamp):
    index = bisect.bisect_right([item[0] for item in samples], timestamp) - 1
    return samples[index][1] if index >= 0 else 0


def time_at(samples, count):
    return next((t for t, n in samples if n >= count), samples[-1][0])


def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'manifests64/decode.json').read_text())
    assert max(len(row['input_ids']) for row in manifest['requests']) <= 512
    state = json.loads((root / 'ar-matrix/state.json').read_text())
    result = []
    for cell in state['results']:
        if cell['phase'] != 'decode' or cell['kind'] != 'measured':
            continue
        data = json.loads((root / 'ar-matrix' / Path(cell['output']).name).read_text())
        assert data['status'] == 'complete'
        wave = data['rounds'][0]['waves'][0]
        requests = wave['requests']
        start = max(time_at(q['samples'], 32) for q in requests)
        end = min(time_at(q['samples'], 1400) for q in requests)
        if end <= start:
            continue
        tokens = sum(count_at(q['samples'], end) - count_at(q['samples'], start)
                     for q in requests)
        result.append(dict(concurrency=data['concurrency'], round=0, wave=0,
                           diagnostic_seconds=end-start,
                           diagnostic_tokens=tokens,
                           diagnostic_tok_s=tokens/(end-start),
                           original_full_wave_tok_s=wave['resident_tok_s'],
                           formal_three_round_median=data['median_decode_tok_s']))
    output = dict(metric='diagnostic_common_window_generated32_to_atmost1400',
                  caveat='Post-hoc first-wave context slice, not a new benchmark or ABBA.',
                  results=result)
    (root / 'context-window-diagnostic.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
