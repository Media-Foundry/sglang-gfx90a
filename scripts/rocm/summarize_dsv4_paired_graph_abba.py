#!/usr/bin/env python3
"""Audit completed same-process ABBA artifacts before reporting timing deltas."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import struct


def geomean(values):
    assert values and all(math.isfinite(v) and v > 0 for v in values)
    return math.exp(statistics.mean(math.log(v) for v in values))


def summarize(path):
    state = json.loads(path.read_text())
    return summarize_state(state)


def summarize_fixed_processes(paths):
    """Four independently validated single-graph launches, never same-process."""
    assert len(paths) == 4
    states = [json.loads(path.read_text()) for path in paths]
    assert len({s.get('attention_issue_order', 0) for s in states}) == 1, 'mixed attention issue order in down ABBA'
    pids = [s['pid'] for s in states]
    assert len(set(pids)) == 4
    blocks = []
    for index, (state, candidate) in enumerate(zip(states, (True, False, False, True))):
        assert state['fixed_warmup_single'] is True
        assert state['status'] == 'complete_pending_c32_hash_review'
        assert state['final_arm'] is candidate
        assert len(state['blocks']) == 1
        block = state['blocks'][0]
        assert block['index'] == 0 and block['candidate'] is candidate
        blocks.append(dict(block, index=index))
    result = summarize_state(dict(status='complete_pending_c32_hash_review',
                                  final_arm=False, blocks=blocks, pid=None))
    result.update(mode='fresh-process fixed-first-use ABBA', pids=pids,
                  state_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    return result


def summarize_state(state):
    assert state['status'] == 'complete_pending_c32_hash_review'
    assert state['final_arm'] is False
    blocks = state['blocks']
    assert len(blocks) == 4 and [b['candidate'] for b in blocks] == [True, False, False, True]
    results, digests = [], {}
    reference_ids, workload = None, None
    for index, block in enumerate(blocks):
        assert block['index'] == index and block['status'] == 'complete'
        artifacts = {}
        for kind in ('c1', 'c32', 'france'):
            file = Path(block[kind])
            raw = file.read_bytes()
            digests[str(file)] = hashlib.sha256(raw).hexdigest()
            artifacts[kind] = json.loads(raw)
        france, c1, c32 = (artifacts[k] for k in ('france', 'c1', 'c32'))
        assert france['exact_count'] == france['request_count'] == 32
        assert c1['status'] == 'complete'
        rows = [r for r in c1['measurements'] if r['rep'] >= 0]
        assert len(rows) == 12
        cases = sorted({r['case'] for r in rows})
        assert len(cases) == 3
        if reference_ids is None:
            reference_ids = {r['case']: r['output_ids'] for r in rows if r['rep'] == 0}
        rates = []
        for case in cases:
            selected = [r for r in rows if r['case'] == case]
            assert sorted(r['rep'] for r in selected) == [0, 1, 2, 3]
            for r in selected:
                ids = r['output_ids']
                assert len(ids) == r['tokens'] == 256
                assert ids == reference_ids[case]
                assert hashlib.sha256(struct.pack('<256I', *ids)).hexdigest() == r['sha256']
            values = sorted(r['tok_s'] for r in selected)
            rates.append(statistics.mean(values[1:-1]))
        assert c32['round_count'] == 6 and c32['request_count'] == 32 and c32['tokens'] == 256
        if workload is None:
            workload = c32['selected_workload_sha256']
        assert c32['selected_workload_sha256'] == workload
        waves = c32['rounds']
        assert len(waves) == 6
        for wave in waves:
            assert wave['lengths'] == [256] * 32
            assert wave['finish_reasons'] == ['length'] * 32
            assert wave['spec_accept_length_mean'] is None
            assert len(wave['output_ids']) == len(wave['completion_sha256']) == 32
            for ids, sha in zip(wave['output_ids'], wave['completion_sha256']):
                assert len(ids) == 256
                assert hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() == sha
        exact = sum(all(wave['output_ids'][i] == waves[0]['output_ids'][i] for wave in waves)
                    for i in range(32))
        results.append(dict(index=index, candidate=block['candidate'],
                            c1_trimmed=geomean(rates),
                            http_warm=statistics.median(w['aggregate_tok_s'] for w in waves[1:]),
                            resident_warm=statistics.median(w['resident_bs32_tok_s'] for w in waves[1:]),
                            c32_cross_round_exact=exact))
    metrics = {}
    for metric in ('c1_trimmed', 'http_warm', 'resident_warm'):
        control = geomean([r[metric] for r in results if not r['candidate']])
        candidate = geomean([r[metric] for r in results if r['candidate']])
        metrics[metric] = dict(control=control, candidate=candidate,
                               delta_pct=100 * (candidate / control - 1))
    return dict(pid=state['pid'], blocks=results, metrics=metrics,
                workload_sha256=workload, artifact_sha256=digests,
                c1_cross_arm_completion_exact=48,
                c32_completion_length_finish_hash_valid=768,
                caveat='C32 hash integrity and France prefix are not proof of full semantic or bitwise parity')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('state', type=Path, nargs='?')
    parser.add_argument('--fixed-process-states', nargs=4, type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert (args.state is None) != (args.fixed_process_states is None)
    result = (summarize_fixed_processes(args.fixed_process_states)
              if args.fixed_process_states is not None else summarize(args.state))
    assert not args.output.exists()
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result['metrics'], indent=2))
