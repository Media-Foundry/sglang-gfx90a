"""Summarize exact Top-K E2E ABBA, including output/logprob parity gates."""
import argparse
import json
import statistics as st
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    arms = ('A1', 'B1', 'B2', 'A2')
    data = {a: {k: json.loads((args.directory / f'{a}_{k}.json').read_text())
                for k in ('c1', 'prefill_c1', 'prefill_c16', 'long_c1', 'long_c16_oracle')}
            for a in arms}
    result = {'arms': {}, 'checks': {}}
    samples = {}
    for arm, d in data.items():
        samples[arm] = {
            'short_c1_http_tok_s': [r['tok_s'] for r in d['c1']['measurements'] if r['rep'] >= 0],
            'prefill_c1_ttft_s': [r['request_ttft_s'][0] for r in d['prefill_c1']['rounds'][1:]],
            'prefill_c16_input_tok_s': [r['aggregate_input_tok_s'] for r in d['prefill_c16']['rounds'][1:]],
            'long_c1_decode_tok_s': [r['request_decode_tok_s'][0] for r in d['long_c1']['rounds'][1:]],
            'long_c1_http_tok_s': [r['aggregate_output_tok_s'] for r in d['long_c1']['rounds'][1:]],
        }
        result['arms'][arm] = {k: {'samples': v, 'median': st.median(v)} for k, v in samples[arm].items()}
    baseline = data['A1']
    for arm, d in data.items():
        short_ids = {r['case']: r['output_ids'] for r in baseline['c1']['measurements'] if r['rep'] == 0}
        forced = lambda x: [{k: v for k, v in r.items() if k != 'wall_s'} for r in x['teacher_forced']]
        c16_fields = ('output_ids', 'output_token_logprobs', 'output_top_logprobs', 'text')
        result['checks'][arm] = {
            'france': d['c1']['france_exact'],
            'short_ids': all(r['output_ids'] == short_ids[r['case']] for r in d['c1']['measurements']),
            'short_teacher_forced': forced(d['c1']) == forced(baseline['c1']),
            'long_c1_ids': all(r['completion_ids'] == baseline['long_c1']['rounds'][0]['completion_ids'] for r in d['long_c1']['rounds']),
            'long_c16_ids_logprobs_text': all(all(r[k] == ref[k] for k in c16_fields)
                for r, ref in zip(d['long_c16_oracle']['rows'], baseline['long_c16_oracle']['rows'])),
            'prefill_c16_ids': all(r['completion_ids'] == baseline['prefill_c16']['rounds'][0]['completion_ids'] for r in d['prefill_c16']['rounds']),
            'zero_cache': all(r['cached_tokens'] == 0 for r in d['long_c16_oracle']['rows']) and
                all(all(v == 0 for v in r['cached_tokens']) for k in ('prefill_c1','prefill_c16','long_c1') for r in d[k]['rounds']),
        }
    result['pooled'] = {}
    for metric in samples['A1']:
        a = samples['A1'][metric] + samples['A2'][metric]
        b = samples['B1'][metric] + samples['B2'][metric]
        result['pooled'][metric] = {'A_median': st.median(a), 'B_median': st.median(b),
            'B_over_A_percent': (st.median(b) / st.median(a) - 1) * 100}
    result['all_checks_pass'] = all(all(v.values()) for v in result['checks'].values())
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.write_text(encoded + '\n')
    print(encoded)


if __name__ == '__main__':
    main()
