"""Post-hoc five-second C64 warmup slices, including batch drain.

This is diagnostic streaming wall time, not a kernel profiler or a replacement
for the formal common-resident metric. Counts include all requests; the
fully_active count only includes requests alive for the whole interval.
"""
import argparse
import bisect
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=root/'ar-matrix/decode_c64_warmup.json')
    parser.add_argument('--output', type=Path, default=root/'long-decode-diagnostic.json')
    args = parser.parse_args()
    path = args.source.resolve()
    data = json.loads(path.read_text())
    assert data['status'] == 'complete'
    wave = data['rounds'][0]['waves'][0]
    requests = wave['requests']
    origin = min(q['samples'][0][0]-q['ttft'] for q in requests)
    for q in requests:
        q['times'] = [t for t, n in q['samples']]

    def count(q, timestamp):
        i = bisect.bisect_right(q['times'], timestamp)-1
        return q['samples'][i][1] if i >= 0 else 0

    rows = []
    for sec in range(10, int(wave['http_wall_seconds'])-4, 10):
        begin, end = origin+sec, origin+sec+5
        active = [q for q in requests if q['times'][0] <= begin and q['times'][-1] >= end]
        rows.append(dict(start_relative_s=sec, duration_s=5, fully_active=len(active),
                         max_generated_among_fully_active=max((count(q, begin) for q in active), default=0),
                         aggregate_tok_s=sum(count(q, end)-count(q, begin) for q in requests)/5))
    result = dict(source=str(path.relative_to(root)), interval_semantics=__doc__,
                  resident_tok_s=wave['resident_tok_s'],
                  http_output_tok_s=wave['http_output_tok_s'], slices=rows)
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
