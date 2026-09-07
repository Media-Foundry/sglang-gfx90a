"""Read TP8 P/D suite artifacts, keeping resident decode separate from HTTP."""
import argparse
import json
from pathlib import Path
import statistics as st


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    result = {'arms': {}, 'oracle_checks': {}}
    baseline = None
    for arm in ('A1', 'B1', 'B2', 'A2'):
        path = args.directory / arm
        if not (path / 'oracle_c32.json').exists():
            continue
        d = {k: json.loads((path / f'{k}.json').read_text()) for k in
            ('decode_c1', 'prefill_c1', 'prefill_c32', 'decode_c32', 'oracle_c32')}
        if baseline is None:
            baseline = d
        samples = {
            'D1_short_HTTP_output_tok_s': [r['tok_s'] for r in d['decode_c1']['measurements'] if r['rep'] >= 0],
            'P1_TTFT_s': [r['prefill_wall_s'] for r in d['prefill_c1']['rounds'][1:]],
            'P1_input_tok_s': [r['aggregate_input_tok_s'] for r in d['prefill_c1']['rounds'][1:]],
            'P32_wave_TTFT_s': [r['prefill_wall_s'] for r in d['prefill_c32']['rounds'][1:]],
            'P32_input_tok_s': [r['aggregate_input_tok_s'] for r in d['prefill_c32']['rounds'][1:]],
            'D32_resident_output_tok_s': [r['resident_bs32_tok_s'] for r in d['decode_c32']['rounds'][1:]],
            'D32_HTTP_output_tok_s': [r['aggregate_tok_s'] for r in d['decode_c32']['rounds'][1:]],
        }
        result['arms'][arm] = {k: {'samples': v, 'median': st.median(v)} for k,v in samples.items()}
        forced = lambda x: [{k:v for k,v in r.items() if k != 'wall_s'} for r in x['teacher_forced']]
        rows, ref = d['oracle_c32']['rows'], baseline['oracle_c32']['rows']
        reference_ids = {r['case']: r['output_ids'] for r in baseline['decode_c1']['measurements'] if r['rep'] == 0}
        result['oracle_checks'][arm] = {
            'france_C1': d['decode_c1']['france_exact'],
            'france_C32': all(r['france_first9_exact'] for r in d['decode_c32']['rounds']),
            'C1_teacher_forced_vs_A1': forced(d['decode_c1']) == forced(baseline['decode_c1']),
            'C1_full_output_vs_A1': all(r['output_ids'] == reference_ids[r['case']] for r in d['decode_c1']['measurements']),
            'P32_cross_round_exact': d['prefill_c32']['cross_round_completion_exact'],
            'D32_cross_round_exact': d['decode_c32']['cross_round_all_exact'],
            'D32_first_divergence': d['decode_c32']['first_divergence_by_request'],
            'fixed_C32_vs_A1': {k:sum(x[k] == y[k] for x,y in zip(rows,ref)) for k in
                ('output_ids','output_token_logprobs','output_top_logprobs','text')},
            'oracle_all_cache_zero': all(r['cached_tokens'] == 0 for r in rows),
            'prefill_all_cache_zero': all(all(v == 0 for v in r['cached_tokens'])
                for k in ('prefill_c1','prefill_c32') for r in d[k]['rounds']),
        }
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
