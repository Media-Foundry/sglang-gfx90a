"""Recompute the completed P baseline and final native-D matrix from evidence."""
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

ROOT = Path('/home/pc/Code/sglang')
EXP = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'scripts/rocm'))
from summarize_dsv4_open_code_matrix import decode_summary, prefill_summary


def load(path):
    return json.loads(path.read_text())


def audited(path, fn):
    data = load(path)
    assert len(data['rounds']) == 3
    result = fn(data)
    result.update(artifact=str(path.relative_to(EXP)),
                  sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    return result


def main():
    baseline = load(EXP/'ar-matrix/state.json')
    final_path = EXP/'ar-final-decode'
    final = load(final_path/'state.json')
    service = load(EXP/'empty-tiles-Final-service.json')
    assert service['pid'] == final['pid'] and service['profile_default']
    for state in (baseline, final):
        assert state['status'] == 'complete'
        assert state['tp'] == 8 and state['mode'] == 'ar'
        assert state['measured_rounds'] == 3
        assert state['concurrencies'] == [1, 2, 4, 8, 16, 32, 64]
        assert '--speculative-algorithm' not in state['command']
        assert state['command'][state['command'].index('--max-total-tokens')+1] == '1048576'
    cells = []
    for c in final['concurrencies']:
        p = audited(EXP/f'ar-matrix/prefill_c{c}_measured.json', prefill_summary)
        a = audited(EXP/f'ar-matrix/decode_c{c}_measured.json', decode_summary)
        b = audited(final_path/f'decode_c{c}_measured.json', decode_summary)
        assert a['manifest_sha256'] == b['manifest_sha256']
        assert all(t >= 30 for t in [r['decode_seconds'] for r in load(final_path/f'decode_c{c}_measured.json')['rounds']])
        cells.append(dict(concurrency=c, prefill=p, baseline_decode=a, final_decode=b,
                          decode_change_pct=(b['median']/a['median']-1)*100))
    arms = []
    for label in ('A1', 'B1', 'B2', 'A2'):
        path = EXP/f'empty-tiles-{label}.json'
        arms.append(dict(label=label, **decode_summary(load(path)),
                         artifact=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    assert len({x['manifest_sha256'] for x in arms}) == 1
    long_gain = (statistics.mean(x['median'] for x in arms if x['label'].startswith('B')) /
                 statistics.mean(x['median'] for x in arms if x['label'].startswith('A')))
    log = (EXP/'empty-tiles-Final.service.log').read_text(errors='replace')
    tiers = sorted({int(x) for x in re.findall(r'Decode graph replay:.*?key_size=(\d+).*?mode=DECODE', log)})
    assert all(c in tiers for c in final['concurrencies']), tiers
    assert 'empty-tile kernel selected' in log
    assert 'max_total_num_tokens=1048576' in log
    stop_time = final_path.joinpath('state.json').stat().st_mtime
    samples = []
    for line in (EXP/'final-vram-snapshots.jsonl').read_text().splitlines():
        row = json.loads(line)
        if row['time'] > stop_time:
            continue
        gpu_data = row['gpus']
        if isinstance(gpu_data, dict):
            gpu_data = gpu_data['gpu_data']
        samples.extend((row['time'], g) for g in gpu_data)
    peaks = []
    for gpu in range(8):
        items = [(t, g['mem_usage']) for t, g in samples if g['gpu'] == gpu]
        assert items
        t, mem = max(items, key=lambda x: x[1]['used_vram']['value'])
        assert mem['used_vram']['unit'] == mem['total_vram']['unit']
        peaks.append(dict(gpu=gpu, time=t, samples=len(items), **mem,
                          percent=100*mem['used_vram']['value']/mem['total_vram']['value']))
    report = dict(tp=8, mode='native_ar', baseline_head=baseline['git_head'],
                  final_head=service['git_head'], final_controller_head=final['git_head'],
                  cells=cells, long_context_abba=arms,
                  long_context_resident_speedup=long_gain, observed_graph_tiers=tiers,
                  final_observed_vram=peaks, final_matrix_end_epoch=stop_time,
                  caveats=[
                      'P: retained completed baseline; exact new guard excludes prefill.',
                      'D: new process, real512-token inputs, natural EOS, at most2048 output tokens.',
                      'Each D round accumulates >=30s common resident windows. Three measured rounds; warmups excluded.',
                      'Baseline vs final D is sequential matrix comparison, not per-tier ABBA. Outputs can differ.',
                      'Long8K C32 comparison is separate A1/B1/B2/A2; do not apply its multiplier to short D.',
                      '1M logical token pool allocated; this is not a1M filled-context or accuracy test.',
                      'P concurrency is client requests; admission16 and chunk36864 constrain actual GPU batch.',
                      'Checkpoint precision unchanged; existing large-prefill BF16-CK is not bitwise SDOT.',
                      'AMD VRAM samples ~5s apart, including startup/final matrix only; not exact allocator peak.',
                      'Exact component score/TopK tests do not establish whole-model bitwise or factual correctness.',
                  ])
    (EXP/'final-report.json').write_text(json.dumps(report, indent=2)+'\n')
    lines = ['# Original DeepSeek V4 Flash: TP8 native AR revalidation', '',
             '| C | P input tok/s | D resident output tok/s | D whole-wave HTTP tok/s |',
             '|--:|--:|--:|--:|']
    for row in cells:
        lines.append(f"| {row['concurrency']} | {row['prefill']['median']:.2f} | "
                     f"{row['final_decode']['median']:.2f} | {row['final_decode']['http_aggregate_tok_s']:.2f} |")
    lines += ['', f'Long8K C32 resident ABBA speedup: {long_gain:.3f}x.', '',
              '## Measurement scope', '', *['- '+x for x in report['caveats']], '']
    (EXP/'RESULTS.md').write_text('\n'.join(lines))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
