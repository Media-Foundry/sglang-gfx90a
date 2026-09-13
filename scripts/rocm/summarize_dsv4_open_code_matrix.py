#!/usr/bin/env python3
"""Recompute open-code P/D rates from saved waves; never infer missing cells."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def checked_ratio(num, seconds, reported):
    assert seconds > 0
    actual = num / seconds
    assert math.isclose(actual, reported, rel_tol=1e-9), (actual, reported)
    return actual


def decode_summary(data):
    assert data['status'] == 'complete' and data['ignore_eos'] is False
    rates, walls, completions, windows = [], [], [], []
    for row in data['rounds']:
        total_tokens, total_seconds = 0, 0.
        for wave in row['waves']:
            count = 0
            for request in wave['requests']:
                ids = request['output_ids']
                assert ids and request['spec_accept_length'] is None
                assert hashlib.sha256(json.dumps(ids).encode()).hexdigest() == request['sha256']
                assert request['samples'][-1][1] == len(ids)
                if wave['resident_seconds'] > 0:
                    def at(t):
                        return next((n for s, n in reversed(request['samples']) if s <= t), 0)
                    count += at(wave['common_end']) - at(wave['common_start'])
            assert count == wave['resident_tokens']
            total_tokens += count
            total_seconds += wave['resident_seconds']
            walls.append(wave['http_wall_seconds'])
            completions.append(sum(len(q['output_ids']) for q in wave['requests']))
            windows.append(wave['resident_seconds'])
        assert math.isclose(total_seconds, row['decode_seconds'])
        assert total_tokens == row['decode_tokens']
        rates.append(checked_ratio(total_tokens, total_seconds, row['decode_tok_s']))
    assert math.isclose(statistics.median(rates), data['median_decode_tok_s'])
    return dict(rates=rates, median=statistics.median(rates),
                http_aggregate_tok_s=sum(completions)/sum(walls),
                waves=len(windows), common_seconds=sum(windows),
                manifest_sha256=data['manifest_sha256'])


def prefill_summary(data):
    rates = []
    for row in data['rounds']:
        assert row['completion_lengths'] == [1] * row['request_count']
        assert not any(row['cached_tokens']), 'P must be fresh-cache'
        rates.append(checked_ratio(row['total_prompt_tokens'], row['prefill_wall_s'],
                                   row['aggregate_input_tok_s']))
    assert math.isclose(statistics.median(rates), data['median_input_tok_s'])
    return dict(rates=rates, median=statistics.median(rates),
                manifest_sha256=data['input_manifest_sha256'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    state = json.loads((args.directory/'state.json').read_text())
    assert state['mode'] == 'ar', 'This auditor intentionally requires native AR'
    result = dict(status=state['status'], tp=state['tp'], pid=state['pid'],
                  git_head=state['git_head'], command=state['command'], cells=[])
    for c in state['concurrencies']:
        cell = {'concurrency':c}
        for phase, fn in [('prefill',prefill_summary),('decode',decode_summary)]:
            path = args.directory/f'{phase}_c{c}_measured.json'
            completed = any(r['phase'] == phase and r['concurrency'] == c
                            and r['kind'] == 'measured' for r in state['results'])
            if not completed:
                cell[phase] = None
                continue
            data = json.loads(path.read_text())
            assert len(data['rounds']) == state['measured_rounds']
            cell[phase] = fn(data)
            cell[phase]['artifact'] = str(path)
            cell[phase]['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        result['cells'].append(cell)
    print('| C | Prefill input tok/s | Native AR resident tok/s |')
    print('|--:|--:|--:|')
    for cell in result['cells']:
        def rate(phase):
            return f"{cell[phase]['median']:.2f}" if cell[phase] else 'pending'
        print(f"| {cell['concurrency']} | {rate('prefill')} | {rate('decode')} |")
    if args.output:
        args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    main()
