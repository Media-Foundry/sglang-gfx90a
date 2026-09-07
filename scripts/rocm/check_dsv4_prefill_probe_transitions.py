#!/usr/bin/env python3
"""Repeat fixed-prefix one-token probes; record the pending request before send.

This checks prefill/request transitions, not cached-decode numerical parity.
No generated code is executed. Requests are sequential and use fresh salts.
"""

import argparse
import json
import math
from pathlib import Path
import time
import urllib.request
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:30011')
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=10)
    parser.add_argument('--timeout', type=float, default=180)
    parser.add_argument('--no-logprobs', action='store_true')
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())['teacher_forced']
    assert len(reference) == 6, 'Expected the established six-probe corpus'
    output = {'reference': str(args.reference), 'logprobs': not args.no_logprobs,
              'status': 'running', 'pending': None, 'responses': []}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def save():
        tmp = args.output.with_suffix(args.output.suffix + '.tmp')
        tmp.write_text(json.dumps(output, indent=2) + '\n')
        tmp.replace(args.output)

    for rep in range(args.rounds):
        for index, source in enumerate(reference):
            output['pending'] = {'round': rep, 'probe': index,
                                 'case': source['case'],
                                 'input_length': len(source['input_ids'])}
            save()
            print('begin', output['pending'], flush=True)
            payload = {'input_ids': source['input_ids'],
                       'sampling_params': {'temperature': 0, 'max_new_tokens': 1},
                       'cache_salt': 'transition-' + uuid.uuid4().hex}
            if not args.no_logprobs:
                payload.update(return_logprob=True, top_logprobs_num=20,
                               logprob_start_len=(len(source['input_ids'])
                                                 - source['continuation_length']))
            req = urllib.request.Request(args.base_url + '/generate',
                                         data=json.dumps(payload).encode(),
                                         headers={'Content-Type': 'application/json'})
            start = time.perf_counter()
            try:
                with opener.open(req, timeout=args.timeout) as response:
                    result = json.load(response)
            except Exception as error:
                output['status'] = 'client_error_server_state_unknown'
                output['error'] = repr(error)
                save()
                raise
            elapsed = time.perf_counter() - start
            meta = result['meta_info']
            assert meta['cached_tokens'] == 0
            assert meta['completion_tokens'] == len(result['output_ids']) == 1
            for value, *_ in meta.get('input_token_logprobs', []):
                assert value is None or math.isfinite(value)
            row = dict(output['pending'], wall_s=elapsed, response=result,
                       next_id_exact=result['output_ids'] == source['output_ids'])
            if not args.no_logprobs:
                row['input_logprobs_exact'] = (meta['input_token_logprobs']
                                              == source['input_token_logprobs'])
                row['output_logprobs_exact'] = (meta['output_top_logprobs']
                                               == source['output_top_logprobs'])
            output['responses'].append(row)
            output['pending'] = None
            save()
            print('done', rep, index, round(elapsed, 4), row['next_id_exact'], flush=True)
    output['status'] = 'complete'
    save()


if __name__ == '__main__':
    main()
