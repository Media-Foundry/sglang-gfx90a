#!/usr/bin/env python3
"""Untimed C1 continuous-decode numeric diagnostic, separate from throughput.

Unlike a max_new_tokens=1 fixed-prefix request, this executes cached decode.
Compare logits only while the two runs have identical input prefixes. Greedy
output agreement is not itself a proof of logit equality or semantic quality.
"""
import argparse
import json
from pathlib import Path
import urllib.request
import uuid


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-url', default='http://127.0.0.1:30011')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--reference', type=Path)
    p.add_argument('--tokens', type=int, default=256)
    a = p.parse_args()
    assert a.tokens >= 2
    root = Path(__file__).resolve().parents[2]
    cases = json.loads((root / '.agents/memory/dsv4_tp8_diverse_32_input_ids.json').read_text())['requests']
    cases = [c for c in cases if c['id'] in ('diverse-03', 'diverse-15', 'diverse-31')]
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    result = {'status': 'running', 'tokens': a.tokens, 'cases': []}
    reference = json.loads(a.reference.read_text()) if a.reference else None
    if reference:
        assert reference['status'] == 'complete' and reference['tokens'] == a.tokens
    for c in cases:
        payload = {'input_ids': c['input_ids'],
                   'sampling_params': {'temperature': 0, 'max_new_tokens': a.tokens, 'ignore_eos': True},
                   'cache_salt': 'decode-logprobs-' + uuid.uuid4().hex,
                   'return_logprob': True, 'logprob_start_len': len(c['input_ids']) - 1,
                   'top_logprobs_num': 20}
        req = urllib.request.Request(a.base_url + '/generate',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with opener.open(req, timeout=180) as response:
            out = json.load(response)
        meta = out['meta_info']
        ids = out['output_ids']
        lp, top = meta['output_token_logprobs'], meta['output_top_logprobs']
        assert len(ids) == len(lp) == len(top) == a.tokens
        assert meta['completion_tokens'] == a.tokens and meta['cached_tokens'] == 0
        assert [v[1] for v in lp] == ids
        row = {'case': c['id'], 'input_ids': c['input_ids'], 'output_ids': ids,
               'output_token_logprobs': lp, 'output_top_logprobs': top, 'text': out['text']}
        if reference:
            old = next(r for r in reference['cases'] if r['case'] == c['id'])
            assert old['input_ids'] == c['input_ids']
            first = next((i for i, (x, y) in enumerate(zip(ids, old['output_ids'])) if x != y), a.tokens)
            # At the first differing output, the input prefix is still equal.
            count = min(first + 1, a.tokens)
            row['comparison'] = {'first_different_output': first if first < a.tokens else None,
                'same_prefix_positions': count,
                'cached_decode_positions': max(count - 1, 0),
                'exact_cached_top20': sum(top[i] == old['output_top_logprobs'][i] for i in range(1, count)),
                'exact_cached_selected_logprob': sum(lp[i] == old['output_token_logprobs'][i] for i in range(1, count))}
        result['cases'].append(row)
        a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(c['id'], row.get('comparison', 'captured'), flush=True)
    result['status'] = 'complete'
    a.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
